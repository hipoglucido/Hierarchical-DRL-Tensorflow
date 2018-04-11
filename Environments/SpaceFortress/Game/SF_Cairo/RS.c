// Might not work!!
// OS X compilation (with GUI):
/* clang -Wall -g  myvars.c TCOL.c DE.c HM.c RS.c -I/usr/local/include/cairo -L/usr/local/lib/ -lcairo `pkg-config --cflags gtk+-3.0` `pkg-config --libs gtk+-3.0`  -o RS -Wno-dangling-else -Wno-switch -D GUI */
// To disable the gui and compile as a library, leave out the GUI switch above. (i.e. remove
// the -u option)
// To run without GTK warnings: (actually running without any error logging to the terminal)
// ./RS 2>/dev/null

/* test graphics 21.2.90 18:00
            definitions */
//#include <stdarg.h>
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


#include "DE_Minimal.h"

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


// I don't think we do anything with this menu stuff
//extern const char *Friend_Menu[3];
//extern const char *Foe_Menu[3];
//extern char *Mine_Indicator;
extern mine_type Mine_Type;




int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

void handle_3()
{
	if((Key==KEY_3)&&(Lastkey!=KEY_3)&&(!(Timing_Flag))) { /* first I keypress */
//		gettimeofday(&intv_t1, NULL);
		intv_t1 = Loop_Counter;
    Timing_Flag=ON;
    Check_Mine_Flag=ON; /* is used by Get_User_Input(cr) */
	}

	if((Key==KEY_3)&&(Lastkey==KEY_3)&&(Timing_Flag)) {   /* second I keypress */
//		gettimeofday(&intv_t2, NULL);
		intv_t2 = Loop_Counter;
		Timing_Flag=OFF;
		Key=0;   /* to enable consecutive double_press */
		/* where with next keypress Lastkey=0 */
		Display_Interval_Flag=ON;  /* is used in main */
   }
}

void Check_Bonus_Input() {
		if(Bonus_Display_Flag==FIRST_BONUS) {
        Bonus_Wasted_Flag=ON;
    }
		else if(Bonus_Display_Flag==SECOND_BONUS)
		{
        if(!Bonus_Wasted_Flag)
				{
            if(Key==KEY_1)
						{
                No_Of_Points_Bonus_Taken++;
                Points=Points+100;
//								Points_Should_Update = 1;
//								Points_Should_Clean = 1;
            }
						//GDK_KEY_2, Get_User_Input() only calls this function when the input is '1' or '2'
						else
						{
                No_Of_Missiles_Bonus_Taken++;
                Missile_Stock=Missile_Stock+50;
                if(Missile_Stock>=100) Missile_Stock=100;
// 								Shots_Should_Update = 1;
//								Shots_Should_Clean = 1;
            }
        Bonus_Display_Flag=NOT_PRESENT;
        Bonus_Granted=ON;
//        Xor_Bonus_Char(rn);    /* erase present $ char */
//				Bonus_Char_Should_Clean = 0;
//        Write_Bonus_Message(cr); /*  Announce_Bonus  */
				}
		}
}


void Get_User_Input()
{
    if (New_Input_Flag) /* new input occured */
    {
      New_Input_Flag=OFF; /* make sure no repetitions on same input */
      switch (Key) {
        case UP:
  		 		Accel_Input=1;        /*   UP    */
  				break;
  			case LEFT:
  				Rotate_Input=-1;      /*   LEFT  */
  				break;
  			case RIGHT:
  				Rotate_Input=1;       /*   RIGHT */
  				break;
        case SPACE:
          New_Missile_Flag=ON;
          break;
        // case KEY_1:
        //   Check_Bonus_Input();
        //   break;
        // case KEY_2:
        //   Check_Bonus_Input();
        //   break;
        // case KEY_3:
        //   handle_3();
        //   break;
      }
    }
    if(Check_Mine_Flag) /* after first press of 3 */
    {
        Check_Mine_Flag=OFF;
        if((Mine_Flag==ALIVE) && (Mine_Type==FRIEND))
    		{
    			Missile_Type=WASTED;
    //        			Show_Mine_Type(cr, Mine_Indicator);
    			Mine_Char = Mine_Indicator;
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


// What does this do?
void Clear_Interval()   /* clear double-press interval */
{
//        int svcolor;
//        int x,y;
//
//        svcolor=getcolor();
//        setcolor(TEXT_COLOR);
//        setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
//        x=Interval_X; y=Data_Line;
//        putimage(x,y,buffer1,COPY_PUT); /* erase garbage */
//        setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);   /* restore gaming area */
//        setcolor(svcolor); /* restore previous color */
}




void Find_Interval()   /* display double-press interval */
{
//    int svcolor;
//    int x,y; // Unused
    int interval;
//		struct timeval tvDiff;
//    interval=Double_Press_Interval=round(((double)(intv_t2-intv_t1)/(double)CLOCKS_PER_SEC)*1000.0); /* in milliseconds */
//		timeval_subtract(&tvDiff, &intv_t2, &intv_t2);
		interval=intv_t2-intv_t1;
//    if((interval<SF_DELAY*20)&&(interval>SF_DELAY)) /* only when interval makes sense */
		if((interval < 20)&&(interval>1)) /* only when interval makes sense */
    {
        if((interval>=Interval_Lower_Limit)&&(interval<=Interval_Upper_Limit)
             &&(Mine_Flag==ALIVE)&&(Mine_Type==FOE))
				{
					printf("Got interval. 👁 💯 💯 💯 \n");
   				 Missile_Type=VS_FOE;   /* rearm missile */
//       		Mine_Char = Mine_Char;
//				 Show_Mine_Type();
//        Update_Interval(cr);
				}
    }
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

		// We don't do anything with menus?
//    Select_Mine_Menus();
    /*
    clrscr(); // From graphics.h or something
    gotoxy(30,5);
    printf("SPACE  FORTRESS ");
    gotoxy(20,15);
    Select_Mine_Menus();
    printf("Your foe mines are:");
    for(int i=0;i<3;i++) printf("    %c",Foe_Menu[i][0]);
    gotoxy(1,24);
    printf("Type any  key to continue ..\n");
    getch();*/
}



// What does this even do in the game [2]
// Tells the player about the bonuses before the game starts
//void Set_Bonus_Message()
//{
////    int size;
////    int svcolor;
////    int x,y;
////
////    svcolor=getcolor();
////    setcolor(TEXT_COLOR);
////    x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
////    y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
////    gprintf(&x,&y,"Bonus");
////    setcolor(svcolor); /* restore previous color */
////
////    size=imagesize(0,0,40,9);        /*length of 5 characters*/
////    buffer2=malloc(size);
////    getimage(x,y,x+40,y+9,buffer2);
////    putimage(x,y,buffer2,XOR_PUT);
////    setcolor(svcolor);
//}


int Generate_Non_Bonus_Char()
{
   int rn;

   do { rn=randrange(0,9); } // Used to be 10, appereantly random(n) generates in (0, n-1)
   while(rn==Bonus_Indication_Index); // I think the only reason for this being here
//	 is the the original random function always returned a number between 0 and n,
//	 while other random fucntions can return a number in between m and n.
   return(rn);
}


void Generate_Resource_Character()
{
    int lastrn;
    static bonus_character lastchar=NON_BONUS; // This is a struct

    if((lastchar==NON_BONUS) && (No_Of_Bonus_Windows<MAX_BONUS_WINDOWS))
		{
   		if(randrange(0,9)<7) /* display first bonus */
     	{
				No_Of_Bonus_Windows++;
				// An index for an array with bonus characters (like '$) of chartype
				rn=Bonus_Indication_Index;
//				Xor_Bonus_Char(rn); // Put the character/image currently passed to graphics
				lastchar=Bonus_Display_Flag=FIRST_BONUS;
				Bonus_Char_Should_Update = 1;
				Bonus_Wasted_Flag=OFF;
			}
			else /* display non_bonus character */
     	{
				lastrn=rn;
				do { rn=Generate_Non_Bonus_Char(); }
				while(rn==lastrn); /* new char is different from last one */
//				Xor_Bonus_Char(rn); // put the image to game
				Bonus_Char_Should_Update = 1;
				lastchar=Bonus_Display_Flag=NON_BONUS;
			}
		}
    else
		{
    	if(lastchar==FIRST_BONUS)
      {
//				Xor_Bonus_Char(rn);
					Bonus_Char_Should_Update = 1;
					lastchar=Bonus_Display_Flag=SECOND_BONUS;
			}
    	else
			{
    		if(lastchar==SECOND_BONUS)
        {
	        rn=Generate_Non_Bonus_Char();
//	        Xor_Bonus_Char(rn);// put the image to gam
					Bonus_Char_Should_Update = 1;
	        lastchar=Bonus_Display_Flag=NON_BONUS;
				}
			}
		}
}


void Handle_Bonus()
{
	if(!Resource_Flag)   /* resource is off */ // What is a resource
  {
		Resource_Off_Counter++;
		// After a counter reaches a threshold, display a resource
    if(Resource_Off_Counter>=No_Resource_Display_Interval)
    {
	    Resource_Flag=ON;
	    Resource_On_Counter=0;
	    Generate_Resource_Character();
    }
  }
	else   /* Resource_Flag=ON; */
	{
		Resource_On_Counter++;
		if(Resource_On_Counter>=Resource_Display_Interval)
		{
			Bonus_Char_Should_Update = 0;
//			Bonus_Char_Should_Clean = 1;
			Resource_Flag=OFF;
			Resource_Off_Counter=0;
			Bonus_Display_Flag=NOT_PRESENT; /* in case bonus is pressed after  */
			if (Bonus_Granted) // If the player did the interval right
			{
//				Write_Bonus_Message();     /* erase bonus message */ // previous message?
				Bonus_Granted=OFF;
			}
		}
	}
}


void SF_iteration()
{
	Loop_Counter++;
	if(Loop_Counter>MAX_LOOPS) {
		Terminal_State = 1;
		#ifdef GUI
		reset_sf();
		#endif
		return;		
	}
	// This was done by processor interupts, but is allowed automatically by GTK
	Get_User_Input();
	// Pauses the game (when the flag is set, continues this loop)
//	while(Freeze_Flag) Get_User_Input();
	Move_Ship();
	Handle_Missile();
	//            if(Sound_Flag>1) Sound_Flag--;
	//            if(Sound_Flag==1) {Sound_Flag--; nosound();}
	// Handle_Mine();
	Test_Collisions();
	Handle_Shell();
	Handle_Fortress();
	

	// if(Display_Interval_Flag) {   /* of double press */
	//     if(Mine_Type==FOE) Find_Interval();
	//     Display_Interval_Flag=OFF;
	// }
	// Accumulate_Data();
	// Handle_Bonus();

//	Score=Points+Velocity+Control+Speed;
	printf("________________\n");
    printf("Ship_X_Pos %d,\nShip_Y_Pos %d, \nShip_Headings %d, \nMissile_X %d, \nMissile_Y %d\n", Ship_X_Pos,Ship_Y_Pos,Ship_Headings, Missile_X, Missile_Y);
    printf("Terminal_State %d\n", Terminal_State);
}


// Does one iteration of the game: either in animation modus or in game event modus.
// The modus is checked by some global flags
// Returns the mode of the iteration, which might unnused?
void game_iteration()
{
	if(!Explosion_Flag && !Jitter_Flag)
	{
		SF_iteration();
	}
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
  GtkWidget *window;
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
