# Needed for connecting to gmail
import smtplib
# Used for composing email
from email.message import EmailMessage
# Module for manipulating dates and times
from datetime import datetime
# access important information
import os


class Message:
    def __init__(self, message, subject, recipient='bmeilingsports@gmail.com'):
        self.message = message
        self.subject = subject
        self.recipient = recipient

        # starting the gmail server connection
        HOST = 'smtp.gmail.com'
        PORT = '587'
        self.server = smtplib.SMTP(host=HOST, port=PORT)
        self.server.starttls()

        # logging into gmail
        self.SENDER = os.environ['email_address']
        email_password = os.environ['email_password']
        self.server.login(self.SENDER, email_password)

        self.send()

    # sends the message with a current date time stamp
    def send(self):
        now = datetime.now()
        time = now.strftime('%H:%M:%S')
        date = now.strftime('%m/%d/%Y')

        msg = EmailMessage()

        msg.set_content(self.message + f"time:{time} date{date}")
        msg['From'] = self.SENDER
        msg['To'] = self.recipient
        msg['Subject'] = self.subject + ' - trade notification'

        self.server.send_message(msg)
        self.server.quit()
