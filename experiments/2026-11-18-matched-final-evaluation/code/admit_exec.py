"""Recheck shared-host resources after runner hashes, before backend startup.

No device allocation or process signals. This reduces a race; it is not a reservation.
"""
import argparse,json,os,sys,time
from pathlib import Path
from run_calibration import gpu_state,available_kib,busy_ticks,TARGET

TIMEOUT_EXIT=75

def eligible(state):
    return (not state['contexts'] and state['free_mib']>=10000
            and state['available_kib']>=34*1024*1024
            and state['busy_cores']<=4)

def sample():
    ticks=os.sysconf('SC_CLK_TCK');start=time.monotonic();before=busy_ticks()
    time.sleep(1)
    cores=(busy_ticks()-before)/ticks/(time.monotonic()-start)
    return dict(gpu_state(),available_kib=available_kib(),busy_cores=cores)

def run(log,argv,seconds=120):
    assert argv and 0<seconds<=120
    deadline=time.monotonic()+seconds
    with Path(log).open('x') as f:
        while True:
            state=sample();admitted=eligible(state)
            # A slow/broken status command must not extend admission past gate budget.
            expired=time.monotonic()>=deadline
            decision='timeout' if expired else 'exec' if admitted else 'wait'
            f.write(json.dumps(dict(monotonic=time.monotonic(),target_uuid=TARGET,
                                   decision=decision,state=state))+'\n');f.flush()
            if expired:return TIMEOUT_EXIT
            if admitted:
                f.close()
                os.execvpe(str(argv[0]),list(map(str,argv)),os.environ.copy())
                raise RuntimeError('exec unexpectedly returned')
            time.sleep(min(2,max(0,deadline-time.monotonic())))

def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    return run(a.log,command)
if __name__=='__main__':sys.exit(main())
