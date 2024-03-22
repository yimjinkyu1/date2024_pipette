#!/bin/bash


MPIGRAPH_LOG_RAW_FILE=$1
SEND_FILE=$2
RECV_FILE=$3 

SEND_TITLE=`cat $MPIGRAPH_LOG_RAW_FILE | grep Send | grep agpu`
SEND_CONTENT=`cat $MPIGRAPH_LOG_RAW_FILE | grep to | grep -v socket | grep -v "unable"`

echo "$SEND_TITLE" > temp.csv
echo "$SEND_CONTENT" >> temp.csv
TEMP_LOG=`cat temp.csv | grep -v Error | grep -v error | sed 's/to//g' | sed 's/\t/,/g' | sed 's/ //g'` 
echo "$TEMP_LOG" > $SEND_FILE
rm -f temp.csv

#receive
RECV_TITLE=`cat $MPIGRAPH_LOG_RAW_FILE | grep Recv | grep agpu`
RECV_CONTENT=`cat $MPIGRAPH_LOG_RAW_FILE | grep from | grep -v socket`
echo "$RECV_TITLE" > temp.csv
echo "$RECV_CONTENT" >> temp.csv
TEMP_LOG=`cat temp.csv | grep -v Error | grep -v error | sed 's/from//g' | sed 's/\t/,/g' | sed 's/ //g'` 
echo "$TEMP_LOG" > $RECV_FILE
rm -f temp.csv


exit 0

