import os
import sys

if __name__ == '__main__':
    os.system("git add *.py")
    print(2)
    os.system("git commit -m " + sys.argv[1])
    print(3)
    os.system("git push")