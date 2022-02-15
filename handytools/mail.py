from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from op import File


class Mail(object):
    def __init__(self, *args):
        super(Mail, self).__init__(*args)
        self.opf = File()

    def send(self, subject, from_email, to_email, pwd, message, files=[]):
        """send email with attachments

        Args:
            subject (str): email title
            from_email (str): sender email
            to_email (str): receiver email
            pwd (str): password of sender email. reference: https://support.google.com/accounts/answer/185833?hl=en
            message (str): email content
            files (list, optional): absolute path list . Defaults to [].

        Examples:
            >>> from handytools.mail import Mail
            >>> mail = Mail()
            >>> from_email = 'xxx'
            >>> to_email = 'xxx'
            >>> pwd = 'xxx' 
            >>> message = 'this is a test email'
            >>> subject = 'test'
            >>> files = ['xxx', 'xxx']
            >>> mail.send(subject, from_email, to_email, pwd, message, files)
        """
        msg = MIMEMultipart()
        msg['Subject'] = subject  # email subject
        msg['From'] = from_email  # sender email
        msg['To'] = to_email  # receiver email
        # message
        text = MIMEText(message)
        msg.attach(text)
        # attachment
        if files:
            for f in files:
                with open(f, "rb") as fil:
                    part = MIMEApplication(
                        fil.read(),
                        Name=os.path.basename(f)
                    )
                part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(
                    f)
                msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(from_email, pwd)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()


if __name__ == '__main__':
    m = Mail()
    subject = "test exmail"
    from_email = "xxxx"
    to_email = "xxx"
    pwd = "xxxx"
    message = "this is a test email"
    m.send(subject, from_email, to_email, pwd, message, files=[])
