import time

def mylog(similar):
    tm = time.localtime()
    strline = f"{tm.tm_year}-{tm.tm_mon}-{tm.tm_mday} {tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}  similar = {similar}\n"
    with open("log.txt", "a") as f:
        f.write(strline)

if __name__ == "__main__":
    mylog(0.93)
    time.sleep(10)
    mylog(0.82)