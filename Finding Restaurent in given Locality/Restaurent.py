import requests
#import urllib.parse

def locu_search(query,key):

    api_key = key
    url = 'https://api.locu.com/v1_0/venue/search/?api_key=' + api_key
    locality = query.replace(' ','%20')
    final_url = url + "&locality=" + locality + "&category=restaurant"
    json_data=requests.get(final_url).json()
    print(json_data)

    formatted_address=json_data['objects'][0]['country']
    print()
    print(formatted_address)

    for item in json_data['objects']:
       print (item['name'], item['phone'])
       
#Main Function
def main():

    key = 'd2e36f1a04204eee891140a7e61387133cd51b1d'
    locu_search('Delhi',key)

main()