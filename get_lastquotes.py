from subprocess import Popen, PIPE, check_output, STDOUT
import datetime
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

with open('\logs\get_lastquotes_error_'+fulltimestamp+'.txt', 'w') as e:
	#post_processing flag is set to true, which checks the ib positions and submits orders
	#this causes problems when user changes orders after the MOC. added sysexit if post processing in moc_live
    #turned back on because you need this for slippage report and commissions
    proc = Popen(['python', 'moc_live.py','0','0','1','0'], stderr=e)
    proc.wait()
    e.flush()

    
    proc = Popen(['python', 'create_board_history.py','1'], stderr=e)
    proc.wait()
    e.flush()
    '''
    proc = Popen(['python', 'create_board_history_rank.py','1'], stderr=e)
    proc.wait()
    e.flush()
    proc = Popen(['python', 'excel_charts.py','1'], stderr=e)
    proc.wait()
    '''
    
