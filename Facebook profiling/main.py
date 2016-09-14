'''
this programe takes 2 parameters, the first is the source folder contains 
input file, the second parameter is the output folder which stores 9500 xml files

for example: run this program like:
python team7.py /data/public-test-data results/week3
the program will read /data/public-test-data/profile/profile.csv 
and output xml files to [current_running_location]/results/week3/
'''
#!/usr/bin/env python

import os
import sys, getopt
import readprofile

def main(argv):
   inputfilefolder = ''
   outputfilefolder = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'tcss555 -i <inputfilefolder> -o <outputfilefolder>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'tcss555 -i <inputfilefolder> -o <outputfilefolder>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfilefolder = arg
      elif opt in ("-o", "--ofile"):
         outputfilefolder = arg
   print 'Input file folder is: ', inputfilefolder
   if(os.path.isdir(inputfilefolder) == False):
      print 'Input folder does not exist'
      sys.exit(2)
   print 'Output file folder is: ', outputfilefolder
   print 'It may take several minutes to get the results...'
   

   model = readprofile.readprofile(inputfilefolder, outputfilefolder)

if __name__ == "__main__":
   main(sys.argv[1:])




