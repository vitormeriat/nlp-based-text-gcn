import datetime


class PrintLog():

    def __init__(self):
        self.output = []

    def print_log(self, msg='', end='\n'):
        now = datetime.datetime.now()
        t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
            + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + \
            ':' + str(now.second).zfill(2)

        lines = msg.split('\n') if isinstance(msg, str) else [msg]

        for line in lines:
            if line == lines[-1]:
                print('[' + t + '] ' + str(line), end=end)
            else:
                print('[' + t + '] ' + str(line))
            self.output.append(str(line)+end)

    def log_history(self):
        return self.output


def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + \
        ':' + str(now.second).zfill(2)

    lines = msg.split('\n') if isinstance(msg, str) else [msg]
    
    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else:
            print('[' + t + '] ' + str(line))
