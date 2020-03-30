import alpaca_trade_api as tradeapi
# For retrieval of environment variables (alpaca keys)
import os
# Efficient array operations
import numpy as np
# Needed for connecting to gmail
import smtplib
# Used for composing email
from email.message import EmailMessage
# Module for manipulating dates and times
from datetime import datetime

api = tradeapi.REST(os.environ['APCA_API_KEY_ID'], os.environ['APCA_API_SECRET_KEY'], os.environ['APCA_API_BASE_URL'])
#
# data = api.polygon.historic_agg_v2('AAPL', 1, 'day', '2020-01-01', '2020-03-15').df
# print(data)




HOST = 'smtp.gmail.com'
PORT = '587'
server = smtplib.SMTP(host=HOST, port=PORT)
server.starttls()
SENDER = 'bmeilingsports@gmail.com'
EMAIL_PASSWORD = 'unleaded,scoring,Carolina0,7toad'
RECIPIENT = 'bryson.meiling@gmail.com'
server.login(SENDER, EMAIL_PASSWORD)

now = datetime.now()
time = now.strftime('%H:%M:%S')
date = now.strftime('%m/%d/%Y')

msg = EmailMessage()

content = 'Hey hows it going?'

msg.set_content(content)
msg['From'] = SENDER
msg['To'] = RECIPIENT
msg['Subject'] = 'ALGO_NAME' + ' - trade notification'

server.send_message(msg)