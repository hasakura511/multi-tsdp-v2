import requests
from time import gmtime, strftime, time, localtime, sleep

def create_system(system, ownerid,apikey):

    url = 'https://api.collective2.com/world/apiv3/createNewSystem'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"site_visibility_status" : 1,
           "startingcapital" : 50000,
           "ownerpersonid" : ownerid,
           "name" : system,
           "creditSystemTimeInSeconds" : 86400,
           "apikey" : apikey
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    print r.text
