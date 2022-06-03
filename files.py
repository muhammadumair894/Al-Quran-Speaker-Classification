import os
from zipfile import ZipFile
'''
https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    For the given path, get the List of all files in the directory tree 
'''


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


dirName = 'C:\\Users\\muham\\PycharmProjects\\Al-Quran_Voice Classification_v3'

# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
Listofzip = []
for elem in listOfFiles:
    #print(elem)
    ex3 = elem[-3:] #getting file extenstion with last 3-char

    if ex3 == "pdf" or  ex3 == "zip":
        Listofzip.append(elem)

for x in Listofzip:
    with ZipFile(x, 'r') as zip:
        zip.extractall()

print('Done')