# -*- coding: utf-8 -*-
import logging
import smtplib
import os
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from fitxf.math.utils.Log import Logging


class SendMail:

    COMMASPACE = ', '

    PORT_SSL = 465
    PORT_SMTP = 587
    GMAIL_SMTP = 'smtp.gmail.com'

    MAX_TOTAL_FILES_SIZE_MB_EMAIL_ATTCH = 25

    MAIL_MODE_SSL  = 'ssl'
    MAIL_MODE_SMTP = 'smtp'
    MAIL_MODE_SMTP_STARTTLS = 'smtp-starttls'

    def __init__(
            self,
            mode             = MAIL_MODE_SMTP,
            mail_server_url  = GMAIL_SMTP,
            mail_server_port = PORT_SMTP,
            logger = None,
    ):
        self.mode = mode
        self.mail_server_url = mail_server_url
        self.mail_server_port = mail_server_port
        self.logger = logger if logger is not None else logging.getLogger()
        self.__init_smtp()
        return

    def __init_smtp(self):
        self.logger.info(
            'Trying to initialize mail server "' + str(self.mail_server_url)
            + '" port ' + str(self.mail_server_port) + ' using mode "' + str(self.mode) + '"...'
        )
        if self.mode == self.MAIL_MODE_SSL:
            # Create a secure SSL context
            # self.context = ssl.create_default_context()
            self.server = smtplib.SMTP_SSL(
                host = self.mail_server_url,
                port = self.mail_server_port,
                # context=self.context
            )
            self.server.ehlo()
        elif self.mode == self.MAIL_MODE_SMTP:
            self.server = smtplib.SMTP(
                host=self.mail_server_url,
                port=self.mail_server_port
            )
            self.server.ehlo()
        else:
            self.server = smtplib.SMTP(
                host = self.mail_server_url,
                port = self.mail_server_port
            )
            self.server.ehlo()
            self.server.starttls()
        self.logger.info(
            'SMTP mode "' + str(self.mode) + '" successfully initialized.'
        )
        return

    def prepare_message(
            self,
            from_addr,
            to_addrs_list,
            subject,
            text,
            files = None,
    ):
        try:
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = SendMail.COMMASPACE.join(to_addrs_list)
            msg['Date'] = formatdate(localtime=True)
            msg['Subject'] = subject

            if text[0]=='<':
                msg.attach(MIMEText(text, 'html'))
            else:
                msg.attach(MIMEText(text, 'plain'))

            files_allowed = self.__attach_file_check_validity_and_size(
                files_attachment_list = files,
                max_total_files_size = SendMail.MAX_TOTAL_FILES_SIZE_MB_EMAIL_ATTCH
            )

            for f in files_allowed or []:
                with open(f, "rb") as fil:
                    part = MIMEApplication(
                        fil.read(),
                        Name = os.path.basename(f)
                    )
                # After the file is closed
                part['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(f)
                msg.attach(part)
            return msg.as_string()
        except Exception as ex:
            errmsg = 'Error creating email message: ' + str(ex)
            self.logger.error(errmsg)
            raise Exception(errmsg)
        #message = """From: %s\nTo: %s\nSubject: %s\n\n%s
        #    """ % (from_addr, ", ".join(to_addrs_list), subject, text)
        #return message

    def __attach_file_check_validity_and_size(
            self,
            files_attachment_list,
            max_total_files_size = MAX_TOTAL_FILES_SIZE_MB_EMAIL_ATTCH
    ):
        if files_attachment_list is None:
            return []

        files_attachment_list_allowed = []

        cum_size_mb = 0.0
        for filepath in files_attachment_list:
            if os.path.isfile(filepath):
                self.logger.info(
                    'Attachment file path "' + str(filepath) + '" OK'
                )
            else:
                self.logger.error(
                    'Invalid attachment file "' + str(filepath) + '", not attaching to email'
                )
                continue

            fsize_bytes = os.path.getsize(filepath)
            fsize_mb = round(fsize_bytes / (1024 * 1024), 2)

            if fsize_mb+cum_size_mb < max_total_files_size:
                files_attachment_list_allowed.append(filepath)
                cum_size_mb += fsize_mb
                self.logger.info(
                    'Appended file "' + str(filepath) + '" as email attachment size ' + str(fsize_mb)
                    + 'MB, total cumulative ' + str(cum_size_mb) + 'MB'
                )
            else:
                self.logger.warning(
                    'File "' + str(filepath) + '" too big ' + str(fsize_mb)
                    + 'MB. Cumulative = ' + str(fsize_mb+cum_size_mb) + ' Not attaching to email'
                )
        return files_attachment_list_allowed

    def send(
            self,
            user,
            password,
            recipients_list,
            message
    ):
        try:
            if password not in [None, '']:
                self.server.login(
                    user = user,
                    password = password
                )
                self.logger.info(
                    'Login for user "' + str(user) + '" successful.'
                )
            else:
                # If no password passed in, no need to do login
                self.logger.warning(
                    'Not doing login for user "' + str(user) + '", no password given "' + str(password) + '"'
                )
            self.server.sendmail(
                from_addr = user,
                to_addrs  = recipients_list,
                msg       = message
            )
            self.logger.info(
                'Message from '+ str(user) + ' to ' + str(recipients_list)
                + ' sent successfully. Closing server..'
            )
            self.server.close()
            self.logger.info(
                'Mail server "' + str(self.mail_server_url) + '" closed'
            )
        except Exception as ex:
            errmsg = 'Exception sending mail from ' + str(user) + ' to ' + str(recipients_list)\
                     + '. Got exception ' + str(ex) + '.'
            self.logger.error(errmsg)
            raise Exception(errmsg)


if __name__ == '__main__':
    user = 'mapktah.naver.com@gmail.com'
    receivers = ['mapktah@mail.ru']
    subject = 'Test mail Python'
    text = 'Test message from Python client (CAI auto alerts)'
    sm = SendMail(
        mode = SendMail.MAIL_MODE_SMTP_STARTTLS,
        mail_server_url = SendMail.GMAIL_SMTP,
        mail_server_port = SendMail.PORT_SMTP,
        logger = Logging.get_default_logger(log_level=logging.INFO, propagate=False),
    )
    message = sm.prepare_message(
        from_addr = user,
        to_addrs_list = receivers,
        subject = subject,
        text = text,
        files = []
    )

    # message = """From: From Kim Bon <kimbon@gmail.com>
    # To: To All
    # Subject: SMTP e-mail test
    #
    # This is a test e-mail message.
    # """

    sm.send(
        user = user,
        password = 'fshg fvzo aelw xath',
        recipients_list = receivers,
        message = message
    )
    exit(0)
