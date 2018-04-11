#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# This script only works with python 2

from os import system
from time import sleep
from random import choice
import sys
import thread


global os

os = sys.platform 


def print_os_error():
	print("Your os '%d' is not supported yet, bye 🌼" % (os) )
	sys.exit(0)

# Sends a key process to the active window
def send_key_press(key, window_id=None):
	# Do OS checking here
	# Initialization happens higher up
	if os.startswith('darwin'): # OS X
		system("cliclick kp:" + key)
	else:
		# Do something with xdotool
		print_os_error()

# Enacts a sequence of random moves on the active window
def randmove(m, window_id=None):
	send_key_press(m, window_id=None)

def one_two_three_go(action, time=3, zzz=1):
	print(action + "\nIn...")
	for i in range(time, 0, -1):
		print(str(i) + "...")
		sleep(zzz)
	print("Go!")

# function that gets called in a background thread that reads in user input, and then 
# dissatifies some condition related to the list argument of this function 
def input_thread(list):
    raw_input("Press any key to continue")
    list.append("None")


def main():
	# (maybe) Constants
	if os.startswith('darwin'): # OS X
		up = "arrow-up"
	#	down = "arrow-down"
		left = "arrow-left"
		right = "arrow-right"
		shoot = "space"
		key_1 = "" 	# Change SF to function key capability to work with this script
		key_2 = "" 
	else:
		up = ""
		left = ""
		right = ""
		shoot = ""
		key_1 = "" 
		key_2 = "" 
		print_os_error()

	keys = [up,left,shoot,key_1,key_2]
	moves = 500 # number of moves to do
	id1 = None
	id2 = None

	alternate = True # parse this as sys arg? 

	if not alternate:	
		if os.startswith('darwin'): # OS X
			one_two_three_go("Switch to the first window")
		else:
			id1 = input("Window ID #1: ")
			id2 = input("Window ID #2: ")

		move_seq = [choice([up,]) for i in range(moves)]
		for m in move_seq:
			randmove(m, id1)

		
	
		# Let the user switch window on OS X	
		if os.startswith('darwin'): # OS X
			system("say klaar er mee!")
			system("beep")
			one_two_three_go("Switch to the second window", zzz=9)

		for m in move_seq:
			randmove(m, id2)
	else: # Do alternate
		list = []
		thread.start_new_thread(input_thread, (list,))
		while not list and moves:
			m = choice([up, right, left])

			# First activate the app for OS X users
			if os.startswith('darwin'):
				system("""osascript -e 'tell app "RS" to activate'""")

			randmove(m, id1)

			if os.startswith('darwin'):
				system("""osascript -e 'tell app "Boxer" to activate'""")

			randmove(m, id2)
			moves -= 1

	print("💯")


main()