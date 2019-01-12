# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import datetime
import smtplib
import time
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
import sys
# sys.path.append('../common/')
# from globalConfig import snake_boduan_receiver, account


retry_count = 3

send_call_url = 'https://ds.goupupupup.com/api/notify/call'
send_sms_url = 'https://ds.goupupupup.com/api/notify/sms'
send_yeecall_url = 'https://ds.goupupupup.com/api/notify/yeecall'
send_wechat_url = 'https://ds.goupupupup.com/api/notify/wechat'


def send_wechat_tpl_msg(first, keyword1,keyword2,keyword3,remark):
    """
        {{first.DATA}}
        服务类型：{{keyword1.DATA}}
        服务状态：{{keyword2.DATA}}
        服务时间：{{keyword3.DATA}}
        {{remark.DATA}}

        内容示例：
        亲爱的用户，您的服务状态发生变更
        服务类型：申请虚拟主机
        服务状态：成功申请
        服务时间：2016/12/26——2017/12/25
        请在后台当前状态下提交下一阶段资料！

    """
    today = datetime.datetime.now().strftime("%Y%m%d")

    headers = {"Content-Type": "application/json"}

    # print(today)
    data = {'u': "YeeCall",
            'p': "YeeCall" + today,
            'first': first,
            'keyword1': keyword1,
            'keyword2': keyword2,
            'keyword3': keyword3,
            'remark': remark}
    for _ in range(retry_count):
        try:
            r = requests.post(send_wechat_url, json=data, headers=headers, timeout=30)
            l = r.json()
            print(l)
            errcode = l['result']
            errmsg = l['description']
            return errcode, errmsg
        except Exception as e:
            traceback.format_exc()
            print(e)

    raise IOError("无法连接")


def for_send_email(strr, url):
    """
    只适用于使用贪吃蛇邮箱想团队发送波段指标邮件
    :param str:
    :param url:
    :return:
    """
    account = {'smtpserver': 'smtp.exmail.qq.com', 'username': 'snakedata@yeecall.com', 'password': 'Sd180910', 'sender': 'snakedata@yeecall.com'}
    snake_boduan_receiver = ['542362275@qq.com', 'snake@yeecall.com', 'qpoi0017@163.com', 'zrh126226@gmail.com']
    account_from = account
    receiver = snake_boduan_receiver
    # receiver = ['huyanghui@yeecall.com']
    # 设置邮件时间 ,subject,context
    today = datetime.date.today()
    today = today.strftime('%Y-%m-%d')
    context_time = today + '  '
    context_header = '波段指标信号通知:\n\n'
    subject = context_time + context_header
    # 构造文字内容
    now_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    context = now_time_str + ' ' + context_header + strr + "\n信息详情:\n" + url
    send_email(account_from, receiver, subject, context)


def send_email(account, receiver, subject, context):
    """
    发送邮件
    :param account: dcit 发件人账户信息；{'smtpserver': 'smtp.exmail.qq.com', 'username': 'snakedata@yeecall.com', 'password': 'Sd180910', 'sender': 'snakedata@yeecall.com'}
    :param receiver: list 收件人邮箱；['huyanghui@yeecall.com']
    :param subject:
    :param context:
    :return:
    """
    # 下面的发件人，收件人是用于邮件传输的。
    smtpserver = account['smtpserver']
    username = account['username']
    password = account['password']
    sender = account['sender']
    msg = MIMEMultipart('mixed')
    msg['Subject'] = subject
    msg['From'] = username+'<'+username+'>'
    msg['To'] = ";".join(receiver)
    text_plain = MIMEText(context, 'plain', 'utf-8')
    msg.attach(text_plain)
    # 发送邮件
    smtp = smtplib.SMTP(smtpserver, 587)
    smtp.set_debuglevel(1)
    smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(sender, receiver, msg.as_string())
    smtp.quit()


def send_call(to):
    """
        电话通知
        Parameters
        ------
          to:string
                    手机号,格式 8613112345678   每次只支持一个号码
          subject:string
                    主题，"12"表示验证码
          content:string
                    内容
        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息

    """

    today = datetime.datetime.now().strftime("%Y%m%d")

    headers = {"Content-Type": "application/json"}

    # print(today)
    data = {'u': "YeeCall",
            'p': "YeeCall" + today,
            'to': to,
            'subject': "12",
            'content': "345678"}
    for _ in range(retry_count):
        try:
            r = requests.post(send_call_url, json=data, headers=headers, timeout=10)
            l = r.json()
            print(l)
            errcode = l['result']
            errmsg = l['description']
            return errcode, errmsg
        except Exception as e:
            traceback.format_exc()
            print(e)

    raise IOError("无法连接")


def send_sms(to, content):
    """
        短信通知
        Parameters
        ------
          to:string
                    手机号,格式 8613112345678  如果是多个号码以逗号分隔
          content:string
                    短信内容
        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息

    """

    today = datetime.datetime.now().strftime("%Y%m%d")

    headers = {"Content-Type": "application/json"}

    # print(today)
    data = {'u': "YeeCall",
            'p': "YeeCall" + today,
            'to': to,
            'subject': "",
            'content': content}
    for _ in range(retry_count):
        try:
            r = requests.post(send_sms_url, json=data, headers=headers, timeout=10)
            l = r.json()
            print(l)
            errcode = l['result']
            errmsg = l['description']
            return errcode, errmsg
        except Exception as e:
            traceback.format_exc()()
            print(e)

    raise IOError("无法连接")


def send_yeecall(to, subject, content):
    """
        发送消息到yeecall群组
        Parameters
        ------
          to:string
                    群组号
          subject:string
                    主题
          content:string
                    内容
        return
        -------
          errcode:int
                    错误码，0-成功  其他取值错误
          errmsg:string
                     错误信息

    """

    today = datetime.datetime.now().strftime("%Y%m%d")

    headers = {"Content-Type": "application/json"}

    # print(today)
    data = {'u': "YeeCall",
            'p': "YeeCall" + today,
            'to': to,
            'subject': subject,
            'content': content}
    for _ in range(retry_count):
        try:
            r = requests.post(send_yeecall_url, json=data, headers=headers, timeout=10)
            l = r.json()
            print(l)
            errcode = l['result']
            errmsg = l['description']
            return errcode, errmsg
        except Exception as e:
            traceback.format_exc()()
            print(e)

    raise IOError("无法连接")


if __name__ == '__main__':
    # errcode, errmsg = send_call("8613560795956")
    # print(errcode,errmsg)

    # errcode, errmsg = send_sms("8613591162709", u"短信测试0927-192000")
    # print(errcode, errmsg)
    #
    # errcode, errmsg = send_yeecall("8613560795956", "仓位通知ok","xxxx现在已经满仓了,ok巴巴爸爸吧")
    # print(errcode, errmsg)

    # for_send_email('test' + '2018-08-08', 'https://ds.goupupupup.com/image/A_BTCusdt_price_30min.png')

    # send_wechat_tpl_msg(u"操作建议55", u"买卖吖吖", u"上涨形态", u"2014年9月29日", u"欢迎观战2")

    print(for_send_email('测试邮件发送', '这是个url'))
