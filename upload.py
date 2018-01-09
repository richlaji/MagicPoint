import os
import sys

if __name__ == '__main__':
    os.system("git add *.py")
    os.system("git add GenerateTestImages/*.py")
    os.system("git add Tool/*.py")
    os.system("git add TraditionalFeatureDetector/*.py")
    print(2)
    print("git commit -m \"" + sys.argv[1] + "\"")
    os.system("git commit -m \"" + sys.argv[1] + "\"")
    print(3)
    os.system("git push")
