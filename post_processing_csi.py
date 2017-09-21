from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

import os
import shutil
src='\ml-tsdp\data\csidata\\v4futures2\\'
check='\ml-tsdp\data\csidata\\v4futures4\\'
dest='\ml-tsdp\data\csidata\\v4futures5\\'
#check_files= [x for x in os.listdir(check) if '.csv' in x.lower()]
copy_files = [x for x in os.listdir(check) if '.csv' in x.lower()]


i=0
for file_name in copy_files:
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        i+=1
        shutil.copy(full_file_name, dest)
        print 'copied', full_file_name, 'to', dest
print len(copy_files), 'source files', i, 'files copied'

with open('\logs\post_processing_csi_error_'+fulltimestamp+'.txt', 'w') as e:
    #f.flush()
    #e.flush()
    proc = Popen(['python', 'vol_adjsize_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'vol_adjsize.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'refresh_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'slip_report_c2.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'slip_report_ib.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'heatmap_futuresCSI.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'heatmap_futuresCSI_multi.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'run_allsystems.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'vol_adjsize.py','1'], stderr=e)
    proc.wait()
    e.flush()
    #in case moc didnt process
    proc = Popen(['python', 'create_board_history.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'create_board_history_rank.py','1'], stderr=e)
    proc.wait()
    e.flush()