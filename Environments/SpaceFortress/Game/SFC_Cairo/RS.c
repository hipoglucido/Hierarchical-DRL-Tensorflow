// OS X compilation (with GUI):
/* clang -Wall -g  myvars.c TCOL.c DE_Minimal.c HM.c RS.c -I/usr/local/include/cairo -L/usr/local/lib/ -lcairo `pkg-config --cflags gtk+-3.0` `pkg-config --libs gtk+-3.0`  -o Control -Wno-dangling-else -Wno-switch -D GUI */
// To disable the gui and compile as a library, leave out the GUI switch above. (i.e. remove
// the -u option)
// To run without GTK warnings: (actually running without any error logging to the terminal)
// ./RS 2>/dev/null

// Linux compilation:
// clang-3.9 -Wall -g -fPIC myvars.c TCOL.c DE_Minimal.c HM.c RS.c `pkg-config --cflags cairo pkg-config --libs cairo pkg-config --cflags gtk+-3.0 pkg-config --libs gtk+-3.0 ` -lm -o Control -Wno-dangling-else -Wno-switch -D GUI


/* test graphics 21.2.90 18:00
            definitions */

#include <stdio.h>
#include <stdlib.h>
//#include <graphics.h>
//#include <process.h>
//#include <bios.h>
//#include <alloc.h>
//#include <dos.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

//#include "myconst.h"
//#include "myvars.h"

//#ifdef GUI // Prefer Minimal shared libray when not using GUI
//#include "DE.h"
//#else
#include "DE_Minimal.h"
//#endif
#include "HM.h"
#include "TCOL.h"
#include "RS.h"

// Dropped in favor of manually defined key constants to be able to compile a fully GTK
// independent game/library/binary
//#include <gdk/gdkkeysyms.h>

// Linux does not have M_PI in math.h for some reason
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

extern mine_type Mine_Type;

void Get_User_Input()
{

    if (New_Input_Flag) /* new input occured */
    {
    	New_Input_Flag=OFF; /* make sure no repetitions on same input */

		switch(Key) {
		#ifdef NO_DIRECTION
			case UP:
		 		Accel_Input=1;        /*   UP    */
				break;
			case LEFT:
				Rotate_Input=-1;      /*   LEFT  */
				break;
			case RIGHT:
				Rotate_Input=1;       /*   RIGHT */
				break;
			case DOWN:
				Accel_Input=-1;        /*   DOWN    */
				break;
		#else
			case UP:
		 		Accel_Input=1;        /*   UP    */
				break;
			case LEFT:
				Rotate_Input=-1;      /*   LEFT  */
				break;
			case RIGHT:
				Rotate_Input=1;       /*   RIGHT */
				break;
		#endif
   		}

	}
}


void ms_sleep(unsigned long miliseconds)
{
	struct timespec tim, tim2;
  tim.tv_sec = 0;
  tim.tv_nsec = miliseconds * 1000000L;
	nanosleep(&tim , &tim2);
}


void Init_Session() {
    One_Game_Loops=One_Game_Duration*60*20;
    Game_Type=SPACE_FORTRESS;
}

void Init_Game()
{
    Score=0;
    Points=0;
    Velocity=0;
    Control=0;
    Speed=0;
    No_Of_Bonus_Intervals=6;
    No_Of_Points_Bonus_Taken=0;
    No_Of_Missiles_Bonus_Taken=0;
    Ship_Damaged_By_Fortress=0;
    Ship_Damaged_By_Mines=0;
    Fortress_Destroyed=0;
    Normal_Game_Termination=0;
    Vulner_Counter=0;
    Last_Missile_Hit=0; /* to measure interval between two consecutive
                    hits of the fortress */
    Ship_Killings_Counter=0;
    Resource_Flag=OFF;
    Resource_Off_Counter=0;
    Bonus_Display_Flag=NOT_PRESENT;   /* in case bonus is pressed
                    after game ends */
    No_Of_Bonus_Windows=0;
    Missile_Stock=100;
}


void SF_iteration()
{
    Get_User_Input();

	// if(Accel_Input)
	// {
	// 	Score = direction_score();
	// }

	// Pauses the game (when the flag is set, continues this loop)
//	while(Freeze_Flag) Get_User_Input();
	Move_Ship();
	//            if(Sound_Flag>1) Sound_Flag--;
	//            if(Sound_Flag==1) {Sound_Flag--; nosound();}
	Handle_Square();
	Test_Collisions();
	// Maybe turn on again at one point? 🙎
//	Accumulate_Data();

//		Score=Points;
	/*
	printf("________________\n");
	printf("Ship_X_Pos %d,\nShip_Y_Pos %d, \nShip_Headings %d, \nSquare_X %d, \nSquare_Y %d\n", Ship_X_Pos,Ship_Y_Pos,Ship_Headings, Square_X, Square_Y);
    printf("Square_Step  %d\n", Square_Step);
    printf("Terminal_State %d\n", Terminal_State);*/
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}



#ifdef GUI
// Translate this to non gui call (i.e. keep some variable set stuff, and calc the time u need
// to wait still in this thingy)
static gboolean on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{
//	printf("Ship pos on draw: %d %d\n", Ship_X_Pos, Ship_Y_Pos);

	// Oddly enough, clipping seems to work different accros surfaces. Therefore it is
	// sometimes wise to set things to always update here. (a clip within a quartz
	// surface erases everything outside of the clipping region)
	unsigned int elapsed_time;
	struct timeval loop_start_time, loop_end_time, loopDiff;
	struct timespec tim;
  tim.tv_sec = 0;

	gettimeofday(&loop_start_time, NULL); // Get the time at this point

	Initialize_Graphics(cr);  // Why is this needed again

	if(Initialized_Graphics == 0)
	{
	  reset_sf();
		Initialized_Graphics = 1;
	}

	clean(cr);
	SF_iteration();

	gettimeofday(&loop_end_time, NULL);
	timeval_subtract(&loopDiff, &loop_end_time, &loop_start_time);
	elapsed_time = round(loopDiff.tv_usec/1000.0);
   if(elapsed_time < SF_DELAY)
	{
	  	tim.tv_nsec = (SF_DELAY-elapsed_time) * 1000000L;
			nanosleep(&tim , NULL);
	}

	update_drawing(cr);
	return FALSE; // Not sure why this should return false
}



gboolean on_key_press(GtkWidget *widget, GdkEventKey *event, gpointer user_data)
{

	set_key(event->keyval); // Only set the key here
  return FALSE;
}

void animation_loop(GtkWidget *darea)
{
	Init_Session();
	Game_Counter=0;

	do
	{   /* loop on number of games here */
		Init_Game();
//		Reset_Screen(cr); // Moved to set initial vals
		Loop_Counter=0;
		do
		{
			gtk_widget_queue_draw(darea);
			while(gtk_events_pending())
			{
    		gtk_main_iteration_do(TRUE); // Redraw the frame
			}
		}
		while(!Restart_Flag&&!End_Flag&&(Loop_Counter < One_Game_Loops));
		Initialized_Graphics = 0;
		Game_Counter++;
//		printf("Died, restart ? %d end? %d \n", Restart_Flag, End_Flag);
		// Close_Graphics(cr); // Not sure if closing is appropiate (it's impossible to close
		// in this part of the program because we don't have acces to GTK cairo context)
	}
	while(!Restart_Flag && !End_Flag);

	// And the clean up here (close graphics)
}

int main(int argc, char *argv[])
{
	Initialized_Graphics = 0;

// Basic GTK initialization
 	 
  GtkWidget *darea;
  gtk_init(&argc, &argv);
  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  darea = gtk_drawing_area_new();
  gtk_container_add(GTK_CONTAINER(window), darea);

  g_signal_connect(G_OBJECT(darea), "draw", G_CALLBACK(on_draw_event), NULL);
  g_signal_connect(window, "destroy", G_CALLBACK(exit), NULL);
  g_signal_connect (G_OBJECT (window), "key_press_event", G_CALLBACK (on_key_press), NULL);

  gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
  gtk_window_set_default_size(GTK_WINDOW(window), WINDOW_WIDTH, WINDOW_HEIGHT);

  gtk_window_set_title(GTK_WINDOW(window), "Space Fortress");
//	gtk_print_context_get_cairo_context();

  gtk_widget_show_all(window);
	animation_loop(darea);

//	stop_drawing(); // GTK handles this I guess

  return 0;
}

#endif
