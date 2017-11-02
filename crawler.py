import urllib.request

webrequest = urllib.request.urlopen('https://www.google.com')
print(webrequest.read())