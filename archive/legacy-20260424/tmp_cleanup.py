import os,signal,subprocess,time 
CURRENT=os.getpid() 
def ids(): 
    cmd=['powershell','-NoProfile','-Command','Get-Process python,raylet,gcs_server,dashboard,dashboard_agent,monitor -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id'] 
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=8) 
    return [int(x) for x in p.stdout.split() if x.strip().isdigit()] 
pids=ids() 
targets=[pid for pid in pids if pid!=CURRENT] 
print('TARGETS',len(targets)) 
for pid in targets: 
    try: os.kill(pid, signal.SIGTERM)  
    except Exception as exc: print('FAIL',pid,type(exc).__name__) 
time.sleep(2) 
try: remaining=ids() 
except Exception as exc: print('RECHECK_FAIL',type(exc).__name__,exc); remaining=[] 
print('REMAINING',len([pid for pid in remaining if pid!=CURRENT])) 
