from datetime import datetime
import calendar

d = datetime.utcnow()
unixtime = calendar.timegm(d.utctimetuple())
print(unixtime)
date_now = datetime.datetime.now()
print(date_now)