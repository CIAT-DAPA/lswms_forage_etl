import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from bs4 import BeautifulSoup



# Constants
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filepath = os.path.join(project_root, 'data.json')
codes_path = os.path.join(project_root, 'codes')
html_file_path = os.path.join(codes_path,'notification.html')


with open(filepath) as f:
    data = json.load(f)


email_list= data["email_list"]


def send_email_html(user_emails, dynamic_header,dynamic_info,dynamic_content,subject):
    smtp_server = data["smtp_server"]
    smtp_port = data["smtp_port"]
    smtp_username = data["smtp_username"]
    smtp_password = data["smtp_password"]


    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = smtp_username
    msg["To"] = ", ".join(user_emails)

    html_content = ""
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as html_file:
          html_content = html_file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    header = soup.find('h1', {'id': 'header'})
    if header:
        header.append(BeautifulSoup(dynamic_header, 'html.parser'))
    info = soup.find('p', {'id': 'info'})
    if info:
        info.append(BeautifulSoup(dynamic_info, 'html.parser'))

    container_ul = soup.find('tr', {'id': 'container'})
    if container_ul:
        container_ul.append(BeautifulSoup(dynamic_content, 'html.parser'))

    html_content = str(soup)

    msg.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, user_emails, msg.as_string())




