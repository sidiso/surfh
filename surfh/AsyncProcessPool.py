'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import psutil
import os
import fnmatch
import queue as Queue
import multiprocessing
import numpy as np
import traceback
import inspect
import signal
from collections import OrderedDict
import glob
import re
import time


#from DDFacet.Other.progressbar import ProgressBar

SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) \
    for n in dir(signal) if n.startswith('SIG') and '_' not in n )

# PID of parent process
parent_pid = os.getpid()

# Exception type for worker process errors
class WorkerProcessError(Exception):
    pass

class Job(object):
    def __init__ (self, job_id, jobitem, singleton=False, when_complete=None):
        self.job_id, self.jobitem, self.singleton, self.when_complete = \
            job_id, jobitem, singleton, (when_complete or (lambda:None))
        self.result = None
        self.complete = False

    def setResult (self, result):
        self.result = result
        self.complete = True
        self.when_complete()

class AsyncProcessPool (object):
    """
    """
    def __init__ (self):
        print("#############################################")
        print("########## Init APP #########################")
        print("#############################################")
        self._started = False
        # init these here so that jobs can be registered
        self._job_handlers = {}
        self._results_map = {}

    def __del__(self):
        self.shutdown()

    def init(self, ncpu=None, affinity=None, parent_affinity=0, num_io_processes=1, verbose=0, pause_on_start=False):
        """
        Initializes an APP.
        Can be called multiple times at program startup

        Args:
            ncpu:
            affinity:
            parent_affinity:
            num_io_processes:
            verbose:

        Returns:

        """
        self.affinity = affinity
        self.verbose = verbose

        self.pause_on_start = pause_on_start

        if isinstance(self.affinity, int):
            self.cpustep = abs(self.affinity) or 1
            maxcpu = psutil.cpu_count() // self.cpustep
            self.ncpu = ncpu or maxcpu
            self.parent_affinity = parent_affinity
        elif isinstance(self.affinity, list):
            if any(map(lambda x: x < 0, self.affinity)):
                raise RuntimeError("Affinities must be list of positive numbers")
            if psutil.cpu_count() < max(self.affinity):
                raise RuntimeError("There are %d virtual threads on this system. Some elements of the affinity map are "
                                   "higher than this. Check parset." % psutil.cpu_count())
            self.ncpu = ncpu or len(self.affinity)
            if self.ncpu != len(self.affinity):
                print("Warning: NCPU does not match affinity list length. Falling back to "
                                          "NCPU=%d" % len(self.affinity))
            self.ncpu = self.ncpu if self.ncpu == len(self.affinity) else len(self.affinity)
            maxcpu = max(self.affinity) + 1  # zero indexed list
            self.parent_affinity = parent_affinity
        elif isinstance(self.affinity, str) and str(self.affinity) == "enable_ht":
            self.affinity = 1
            self.cpustep = 1
            maxcpu = psutil.cpu_count() // self.cpustep
            self.ncpu = ncpu or maxcpu
            self.parent_affinity = parent_affinity
        elif isinstance(self.affinity, str) and str(self.affinity) == "disable_ht":
            # this works on Ubuntu so possibly Debian-like systems, no guarantees for the rest
            # the mapping on hyperthread-enabled NUMA machines with multiple processors can get very tricky
            # /sys/devices/system/cpu/cpu*/topology/thread_siblings_list should give us a list of siblings
            # whereas core_siblings_list will give the list of sibling threads on the same physical processor
            # for now lets not worry about assigning different jobs to physical cpus but for now keep things
            # simple
            hyperthread_sibling_lists = map(lambda x: x + "/topology/thread_siblings_list",
                                            filter(lambda x: re.match(r"cpu[0-9]+", os.path.basename(x)),
                                                   glob.glob("/sys/devices/system/cpu/cpu*")))
            left_set = set([])
            right_set = set([])
            for siblings_file in hyperthread_sibling_lists:
                with open(siblings_file) as f:
                    siblings = map(int, f.readline().split(","))
                    if len(siblings) == 2:
                        l, r = siblings
                        # cannot be sure the indices don't swap around at some point
                        # since there are two copies of the siblings we need to
                        # check that the items are not in the other sets before adding
                        # them to the left and right sets respectively
                        if l not in right_set:
                            left_set.add(l)
                        if r not in left_set:
                            right_set.add(r)
                    elif len(siblings) == 1:
                        left_set.add(siblings[0])
                    else:
                        raise RuntimeError("Don't know how to handle this architecture. It seems there are more than "
                                           "two threads per core? Try setting things manually by specifying a list "
                                           "to the affinity option")

            self.affinity = list(left_set)  # only consider 1 thread per core
            self.ncpu = ncpu or len(self.affinity)
            if self.ncpu > len(self.affinity):
                print("Warning: NCPU is more than the number of physical cores on "
                                          "the system. I will only use %d cores." % len(self.affinity))
            self.ncpu = self.ncpu if self.ncpu <= len(self.affinity) else len(self.affinity)
            maxcpu = max(self.affinity) + 1  # zero indexed list

            unused = [x for x in range(psutil.cpu_count()) if x not in self.affinity]
            if len(unused) == 0:
                print("No unassigned vthreads to use as parent IO thread, I will use thread 0")
                self.parent_affinity = 0 # none unused (HT is probably disabled BIOS level)
            else:
                self.parent_affinity = unused[0] # grab the first unused vthread
        elif isinstance(self.affinity, str) and str(self.affinity) == "disable":
            self.affinity = None
            self.parent_affinity = None
            self.cpustep = 1
            maxcpu = psutil.cpu_count()
            self.ncpu = ncpu or maxcpu
        else:
            raise RuntimeError("Invalid option for Parallel.Affinity. Expected cpu step (int), list, "
                               "'enable_ht', 'disable_ht', 'disable'")
        if self.parent_affinity is None:
            print("Parent and I/O affinities not specified, leaving unset")
        else:
            print("Fixing parent process to vthread %d" % self.parent_affinity)
        psutil.Process().cpu_affinity(range(self.ncpu) if not self.parent_affinity else [self.parent_affinity])

        # if NCPU is 0, set to number of CPUs on system
        if not self.ncpu:
            self.ncpu = maxcpu
        elif self.ncpu > maxcpu:
            print("NCPU=%d is too high for this setup (%d cores, affinity %s)" %
                                    (self.ncpu, psutil.cpu_count(),
                                     str(self.affinity) if isinstance(self.affinity, int)
                                     else ",".join(map(str, self.affinity)) if isinstance(self.affinity, list)
                                     else "disabled"))
            print("Falling back to NCPU=%d" % (maxcpu))
            self.ncpu = maxcpu
        self.procinfo = psutil.Process()  # this will be used to control CPU affinity

        # create a queue for compute-bound tasks
        # generate list of CPU cores for workers to run on
        if isinstance(self.affinity, int) and (not self.affinity or self.affinity == 1):
            cores = range(self.ncpu)
        elif isinstance(self.affinity, int) and self.affinity == 2:
            cores = range(0, self.ncpu * 2, 2)
        elif isinstance(self.affinity, int) and self.affinity == -2:
            cores = range(0, self.ncpu * 2, 4) + range(1, self.ncpu * 2, 4)
        elif isinstance(self.affinity, list):
            cores = self.affinity[:self.ncpu]
        elif not self.affinity:
            cores = range(self.ncpu)
        else:
            raise ValueError("unknown affinity setting")
        if not self.affinity:
            print("Worker affinities not specified, leaving unset")
        else:
            print("Worker processes fixed to vthreads %s" % (','.join([str(x) for x in cores])),
                                      )
        self._compute_workers = []
        self._io_workers = []
        self._compute_queue   = multiprocessing.Queue()
        self._io_queues       = [ multiprocessing.Queue() for x in range(num_io_processes) ]
        self._result_queue    = multiprocessing.Queue()
        self._termination_event = multiprocessing.Event()
        # this event is set when all workers have been started, an cleared when a restart is requested
        self._workers_started_event = multiprocessing.Event()

        self._cores = cores

        # create a Taras Bulba process. http://www.imdb.com/title/tt0056556/quotes
        # This is responsible for spawning, killing, and respawning workers
        self._taras_restart_event = multiprocessing.Event()
        self._taras_exit_event = multiprocessing.Event()
        if self.ncpu > 1:
            self._taras_bulba = multiprocessing.Process(target=AsyncProcessPool._startBulba, name="TB", args=(self,))
            if pause_on_start:
                print("Please note that due to your debug settings, worker processes will be paused on startup. Send SIGCONT to all processes to continue.")
        else:
            self._taras_bulba = None

        self._started = False

    def registerJobHandlers (self, *handlers):
        """Adds recognized job handlers. Job handlers may be functions or objects."""
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._started:
            raise RuntimeError("Workers already started. This is a bug.")
        for handler in handlers:
            if not inspect.isfunction(handler) and not isinstance(handler, object):
                raise RuntimeError("Job handler must be a function or object. This is a bug.")
            self._job_handlers[id(handler)] = handler


    def startWorkers(self):
        """Starts worker threads. All job handlers and events must be registered *BEFORE*"""
        if self.ncpu > 1:
            self._taras_bulba.start()
        self._started = True

    def _checkResultQueue(self):
        """
        Check the result queue for any unread results, read them off and move them to the result map.
        Return number of results collected.
        """
        nres = 0
        while True:
            try:
                result = self._result_queue.get(False)
            except Queue.Empty:
                return nres
            nres += 1
            # ok, dispatch the result
            job_id = result["job_id"]
            job = self._results_map.get(job_id)
            if job is None:
                raise KeyError("Job '%s' was not enqueued. This is a logic error." % job_id)
            job.setResult(result)

    def restartWorkers(self):
        if self.ncpu > 1:
            self._workers_started_event.clear()
            # place a poison pill onto every queue
            if self.verbose:
                print("asking worker processes to restart")
            for core in self._cores:
                self._compute_queue.put("POISON-E")
            for queue in self._io_queues:
                queue.put("POISON-E")
            if self.verbose:
                print("poison pills enqueued")
            self._taras_restart_event.set()
            nres = self._checkResultQueue()
            if nres:
                print("collected %d outstanding results from the queue"%nres)

    def awaitWorkerStart(self):
        if self.ncpu > 1:
            while not self._workers_started_event.is_set():
                if self._termination_event.is_set():
                    if self.verbose > 1:
                        print("termination event spotted, exiting")
                    raise WorkerProcessError()
                nres = self._checkResultQueue()
                if nres:
                    print("collected %d outstanding results from the queue" % nres)
                print("waiting for worker processes to start up")
                self._workers_started_event.wait(10)

    def _startBulba (self):
        """This runs the Taras Bulba process. A Taras Bulba spawns and kills worker processes on demand.
        The reason for killing workers is to work around potential memory leaks. Since a Bulba is forked
        from the main process early on, it has a very low RAM footprint, so re-forking the workers off
        a Bulba every so often makes sure their RAM usage is reset."""
        try:
            # loop until the completion event is raised
            # at this stage the workers are dead (or not started)
            while not self._taras_exit_event.is_set():
                if self.verbose:
                    print("(re)creating worker processes")
                # create the workers
                self._compute_workers = []
                self._io_workers = []
                for i, core in enumerate(self._cores):
                    proc_id = "comp%02d" % i
                    self._compute_workers.append(
                        multiprocessing.Process(name=proc_id, target=self._start_worker,
                                                args=(self, proc_id, [core], self._compute_queue,
                                                      self.pause_on_start)))
                for i, queue in enumerate(self._io_queues):
                    proc_id = "io%02d" % i
                    self._io_workers.append(
                        multiprocessing.Process(name=proc_id, target=self._start_worker,
                                                args=(self, proc_id, None, queue, self.pause_on_start)))

                # start the workers
                if self.verbose:
                    print("starting  worker processes")
                worker_map = {}
                for proc in self._compute_workers + self._io_workers:
                    proc.start()
                    worker_map[proc.pid] = proc
                dead_workers = {}

                # set event to indicate workers are started
                self._workers_started_event.set()

                # go to sleep until we're told to do the whole thing again
                while not self._taras_restart_event.is_set():
                    if self.verbose:
                        print("waiting for restart signal")
                    try:
                        self._taras_restart_event.wait(5)
                        if self.verbose:
                            print("wait done")
                    except KeyboardInterrupt:
                        print("Ctrl+C caught, exiting")
                        self._termination_event.set()
                        self._taras_exit_event.set()
                    # check for dead children, unless workers_started event has been cleared by restartWorkers()
                    # (in which case they're already going to be exiting)
                    if self._workers_started_event.is_set():
                        for pid, proc in getattr(worker_map, "iteritems", worker_map.items)():
                            if not proc.is_alive():
                                proc.join()
                                dead_workers[proc.pid] = proc
                                if proc.exitcode < 0:
                                    print("worker '%s' killed by signal %s" % (proc.name, SIGNALS_TO_NAMES_DICT[-proc.exitcode]))
                                else:
                                    print("worker '%s' died with exit code %d"%(proc.name, proc.exitcode))
                        # if workers have died, initiate bailout
                        if dead_workers:
                            print("%d worker(s) have died. Initiating shutdown."%len(dead_workers))
                            self._taras_restart_event.set()  # to break out of loop
                            self._termination_event.set()
                            self._taras_exit_event.set()
                self._taras_restart_event.clear()
                if self._termination_event.is_set():
                    if self.verbose:
                        print("terminating workers, since termination event is set")
                    for proc in getattr(worker_map, "itervalues", worker_map.values)():
                        if proc.is_alive():
                            proc.terminate()
                if self.verbose:
                    print("reaping workers")
                # join processes
                for pid, proc in getattr(worker_map, "iteritems", worker_map.items)():
                    if self.verbose:
                        print("reaping worker %d"%pid)
                    proc.join()
                    if self.verbose:
                        print("worker %d's immortal soul has been put to rest"%pid)

                if self.verbose:
                    print("all workers have been reaped")
            if self.verbose:
                print("exiting")
        except:
            print("exception raised in Taras Bulba process, see below. This is a bug!")
            print(traceback.format_exc())
            self._workers_started_event.set()
            self._termination_event.set()
            self._taras_exit_event.set()

    def runJob (self, job_id, handler=None, io=None, args=(), kwargs={},
                event=None, counter=None,
                singleton=False, collect_result=True,
                serial=False):
        """
        Puts a job on a processing queue.

        Args:
            job_id:  string job identifier
            handler: function previously registered with registerJobHandler, or bound method of object that was registered.
            io:     if None, job is placed on compute queues. If 0/1/..., job is placed on an I/O queue of the given level
            event:  if not None, then the named event will be raised when the job is complete.
                    Otherwise, the job is a singleton, handled via the events directory.
            counter: if set to a JobCounter object, the job will be associated with a job counter, which will be incremented upon runJob(),
                    and decremented when the job is complete.
            collect_result: if True, job's result will be collected and returned via awaitJobResults().
                    This mode is only available in the parent process.
            singleton: if True, then job is a one-off. If collect_result=True, then when complete, its result will remain
                    in the results map forever, so that subsequent calls to awaitJobResults() on it return that result.
                    A singleton job can't be run again.
                    If False, job result will be collected by awaitJobResults() and removed from the map: the job can be
                    run again.
            serial: if True, job is run serially in the main process. Useful for debugging.
        """
        if collect_result and os.getpid() != parent_pid:
            raise RuntimeError("runJob() with collect_result can only be called in the parent process. This is a bug.")
        if collect_result and job_id in self._results_map:
            raise RuntimeError("Job '%s' has an uncollected result, or is a singleton. This is a bug."%job_id)
        # make sure workers are started
        self.awaitWorkerStart()
        # figure out the handler, and how to pass it to the queue
        # If this is a function, then describe it by function id, None
        if inspect.isfunction(handler):
            handler_id, method = id(handler), None
            handler_desc  = "%s()" % handler.__name__
        # If this is a bound method, describe it by instance id, method_name
        elif inspect.ismethod(handler):
            instance = getattr(handler, "im_self", handler.__self__)
            if instance is None:
                raise RuntimeError("Job '%s': handler %s is not a bound method. This is a bug." % (job_id, handler))
            handler_id, method = id(instance), handler.__name__
            handler_desc = "%s.%s()" % (getattr(handler, "im_class", handler.__class__).__name__, method)
        else:
            raise TypeError("'handler' argument must be a function or a bound method")

        # create the job item
        jobitem = dict(job_id=job_id, handler=(handler_id, method, handler_desc),
                       counter=counter and id(counter),
                       collect_result=collect_result,
                       args=args, kwargs=kwargs)
        # insert entry into dict of pending jobs
        if collect_result:
            job = self._results_map[job_id] = Job(job_id, jobitem, singleton=singleton)
        ## normal paralell mode, stick job on queue
        if self.ncpu > 1 and not serial:
            print("PARALLEL")
            print("JobITEM is", jobitem)
            if self.verbose > 2:
                print("enqueueing job %s: %s"%(job_id, handler_desc))
            # place it on appropriate queue
            if io is None:
                self._compute_queue.put(jobitem)
            else:
                io = max(len(self._io_queues)-1, io)
                self._io_queues[io].put(jobitem)
        # serial mode: process job in this process, and raise any exceptions up
        else:
            print("SERIAL")
            self._dispatch_job(jobitem, reraise=True)

    def awaitJobCounter (self, counter, progress=None, total=None, timeout=10):
        if self.verbose > 2:
            print("  %s is complete" % counter.name)
        if progress:
            current = counter.getValue()
            total = total or current or 1
            #pBAR = ProgressBar(Title="  "+progress)
            #pBAR.disable()
            #pBAR.render(total-current,total)
            while current:
                current = counter.awaitZeroWithTimeout(timeout)
                #pBAR.render(total - current, total)
                if self._termination_event.is_set():
                    if self.verbose > 1:
                        print("  termination event spotted, exiting")
                    raise WorkerProcessError()
        else:
            counter.awaitZero()
            if self._termination_event.is_set():
                if self.verbose > 1:
                    print("  termination event spotted, exiting")
                raise WorkerProcessError()
            if self.verbose > 2:
                print("  %s is complete" % counter.name)


    def awaitJobResults (self, jobspecs, progress=None, timing=None):
        """
        Waits for job(s) given by arguments to complete, and returns their results.
        Note that this only works for jobs scheduled by the same process, since each process has its own results map.
        A process will block indefinitely is asked to await on jobs scheduled by another process.

        Args:
            jobspec: a job spec, or a list of job specs. Each spec can contain a wildcard e.g. "job*", to wait for
                multiple jobs.
            progress: if True, a progress bar with that title will be rendered
            timing: if True, a timing report with that title will be printed (note that progress implies timing)

        Returns:
            a list of results. Each entry is the result returned by the job (if no wildcard), or a list
            of results from each job (if has a wildcard)
        """
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if type(jobspecs) is str:
            jobspecs = [ jobspecs ]
        # make a dict of all jobs still outstanding
        awaiting_jobs = {}  # this maps job_id to a set of jobspecs (if multiple) that it matches
        job_results = OrderedDict()   # this maps jobspec to a list of results
        total_jobs = complete_jobs = 0
        for jobspec in jobspecs:
            matching_jobs = [job_id for job_id in getattr(self._results_map, "iterkeys", self._results_map.keys)() if fnmatch.fnmatch(job_id, jobspec)]
            for job_id in matching_jobs:
                awaiting_jobs.setdefault(job_id, set()).add(jobspec)
            if not matching_jobs:
                raise RuntimeError("no pending jobs matching '%s'. This is probably a bug." % jobspec)
            total_jobs += len(matching_jobs)
            job_results[jobspec] = len(matching_jobs), []
        # check dict of already returned results (perhaps from previous calls to awaitJobs). Remove
        # matching results, and assign them to appropriate jobspec lists
        delmap = []
        deljobs = []
        for job_id, job in self._results_map.items():
            if job_id in awaiting_jobs and job.complete:
                for jobspec in awaiting_jobs[job_id]:
                    job_results[jobspec][1].append(job.result)
                    complete_jobs += 1
                if not job.singleton:
                    delmap.append(job_id)
                deljobs.append(job_id)
        for job_id in delmap:
            del self._results_map[job_id]
        for job_id in deljobs:
            del awaiting_jobs[job_id]
        #if progress:
        #    pBAR = ProgressBar(Title="  "+progress)
        #    pBAR.render(complete_jobs,(total_jobs or 1))
        if self.verbose > 1:
            print("checking job results: %s (%d still pending)"%(
                ", ".join(["%s %d/%d"%(jobspec, len(results), njobs) for jobspec, (njobs, results) in getattr(job_results, "iteritems", job_results.items)()]),
                len(awaiting_jobs)))
        # sit here while any pending jobs remain
        while awaiting_jobs and not self._termination_event.is_set():
            try:
                result = self._result_queue.get(True, 10)
            except Queue.Empty:
                # print>> log, "checking for dead workers"
                # shoot the zombie process, if any
                multiprocessing.active_children()
                continue
            # ok, dispatch the result
            job_id = result["job_id"]
            job = self._results_map.get(job_id)
            if job is None:
                raise KeyError("Job '%s' was not enqueued. This is a logic error." % job_id)
            job.setResult(result)
            # if being awaited, dispatch appropriately
            if job_id in awaiting_jobs:
                for jobspec in awaiting_jobs[job_id]:
                    job_results[jobspec][1].append(result)
                    complete_jobs += 1
                if not job.singleton:
                    del self._results_map[job_id]
                del awaiting_jobs[job_id]
                #if progress:
                #    pBAR.render(complete_jobs,(total_jobs or 1))
            # print status update
            if self.verbose > 1:
                print("received job results %s" % " ".join(["%s:%d"%(jobspec, len(results)) for jobspec, (_, results)
                                                             in getattr(job_results, "iteritems", job_results.items)()]))
        # render complete
        #if progress:
        #    pBAR.render(complete_jobs,(total_jobs or 1))

        if self._termination_event.is_set():
            if self.verbose > 1:
                print("  termination event spotted, exiting")
            raise WorkerProcessError()

        # process list of results for each jobspec to check for errors
        for jobspec, (njobs, results) in getattr(job_results, "iteritems", job_results.items)():
            num_errors = len([res for res in results if not res['success']])
            #if timing or progress:
            #    print("%s: %d jobs complete, average single-core time %.2fs per job" % (timing or progress, len(results), times.mean()))
            if num_errors:
                print("%s: %d jobs returned an error. Aborting."%(jobspec, num_errors))
                raise RuntimeError("some distributed jobs have failed, and res is ", results)
        # return list of results
        result_values = []
        for jobspec, (_, results) in getattr(job_results, "iteritems", job_results.items)():
            resvals = [resitem["result"] if resitem["success"] else resitem["error"] for resitem in results]
            if '*' not in jobspec:
                resvals = resvals[0]
            result_values.append(resvals)
        return result_values[0] if len(result_values) == 1 else result_values

    def terminate(self):
        if self._started:
            self._termination_event.set()
            # wake up Taras to kill workers
            self._taras_exit_event.set()
            self._taras_restart_event.set()

    def shutdown(self):
        """Terminate worker threads"""
        if not self._started:
            return
        if not self._termination_event.is_set():
            if self.verbose > 1:
                print("shutdown: asking TB to stop workers")
            self._started = False
            self._taras_exit_event.set()
            self.restartWorkers()
        if self._taras_bulba:
                if self.verbose > 1:
                    print("shutdown: waiting for TB to exit")
                self._taras_bulba.join()
        if self.verbose > 1:
            print("shutdown: closing queues")
        # join and close queues
        self._result_queue.close()
        self._compute_queue.close()
        for queue in self._io_queues:
            queue.close()
        if self.verbose > 1:
            print("shutdown complete")

    @staticmethod
    def _start_worker (object, proc_id, affinity, worker_queue, pause_on_start=False):
        """
            Helper method for worker process startup. ets up affinity, and calls _run_worker method on
            object with the specified work queue.

        Args:
            object:
            proc_id:
            affinity:
            work_queue:

        Returns:

        """
        if pause_on_start:
            os.kill(os.getpid(), signal.SIGSTOP)
        AsyncProcessPool.proc_id = proc_id
        if affinity:
            psutil.Process().cpu_affinity(affinity)
        object._run_worker(worker_queue)
        if object.verbose:
            print("exiting worker pid %d"%os.getpid())


    def _dispatch_job(self, jobitem, reraise=False):
        """Handles job described by jobitem dict.

        If reraise is True, any eceptions are re-raised. This is useful for debugging."""
        counter = None
        try:
            job_id, counter_id, args, kwargs = [jobitem.get(attr) for attr in
                                                         ["job_id", "counter", "args", "kwargs"]]
            handler_id, method, handler_desc = jobitem["handler"]
            handler = self._job_handlers.get(handler_id)
            if handler is None:
                raise RuntimeError("Job %s: unknown handler %s. This is a bug." % (job_id, handler_desc))
            # find counter object, if specified
            # instantiate SharedDict arguments
            args = [ arg for arg in args ]

            # call the job
            if self.verbose > 1:
                print("job %s: calling %s" % (job_id, handler_desc))
            if method is None:
                # call object directly
                result = handler(*args, **kwargs)
            else:
                call = getattr(handler, method, None)
                if not callable(call):
                    raise KeyError("Job %s: unknown method '%s' for handler %s" % (job_id, method, handler_desc))
                result = call(*args, **kwargs)
            if self.verbose > 3:
                print("job %s: %s returns %s" % (job_id, handler_desc, result))
            # Send result back
            if jobitem['collect_result']:
                self._result_queue.put(
                    dict(job_id=job_id, proc_id=self.proc_id, success=True, result=result))
        except KeyboardInterrupt:
            raise
        except Exception as exc:

            if reraise:
                raise

            if jobitem['collect_result']:
                self._result_queue.put(
                    dict(job_id=job_id, proc_id=self.proc_id, success=False, error=exc))
        finally:
            if counter is not None:
                counter.decrement()

    def _run_worker (self, queue):
        """
            Runs worker loop on given queue. Waits on queue, picks off job items, looks them up in context table,
            calls them, and returns results in the work queue.
        """
        if self.verbose > 0:
            print("started worker pid %d"%os.getpid())
        try:
            pill = True
            # While no poisoned pill has been given grab items from the queue.
            while pill:
                try:
                    # Get queue item, or timeout and check if pill perscribed.
                    #print>>log,"%s: calling queue.get()"%AsyncProcessPool.proc_id
                    jobitem = queue.get(True, 10)
                    #print>>log,"%s: queue.get() returns %s"%(AsyncProcessPool.proc_id, jobitem)
                except Queue.Empty:
                    continue
                if jobitem == "POISON-E":
                    if self.verbose:
                        print("got pill. Qin:{} Qout:{}".format(queue.qsize(), self._result_queue.qsize()))
                    break
                elif jobitem is not None:
                    self._dispatch_job(jobitem)

        except KeyboardInterrupt:
            print("Ctrl+C caught, exiting")
            return
    # CPU id. This will be None in the parent process, and a unique number in each worker process
    proc_id = None

APP = None

def _init_default():
    global APP
    if APP is None:
        APP = AsyncProcessPool()
        APP.init(psutil.cpu_count(), affinity=0, num_io_processes=1, verbose=0)

_init_default()

def init(ncpu=None, affinity=None, parent_affinity=0, num_io_processes=1, verbose=0, pause_on_start=False):
    global APP
    APP.init(ncpu, affinity, parent_affinity, num_io_processes, verbose, pause_on_start=pause_on_start)


