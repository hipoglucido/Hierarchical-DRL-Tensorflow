// Ubuntu upstream clang is not at 3.9 yet, so alias clang to clang3.9 on Ubuntu
/****************************************** COMPILE THINGY *************************************************

|** Compile a shared library **|
# You can copy/paste all of this 😘
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o aim_frame_lib.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3;
#*** Add -D GUI_INTERFACE to enable acces to full size, full color renders of the game ***#
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3 -D GUI_INTERFACE;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o aim_frame_lib_FULL.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 -D GUI_INTERFACE;


|** Compile a playable GUI version **|
clang -Wall -g -fPIC myvars.c TCOL.c DE_Minimal.c HM.c RS.c `pkg-config --cflags cairo pkg-config --libs cairo pkg-config --cflags gtk+-3.0 pkg-config --libs gtk+-3.0 ` -lm -o Control -Wno-dangling-else -Wno-switch -D GUI

-- Switches:
-D GUI_INTERFACE ** Full sized and colored game renders **
-D GRID_MOVEMENT ** Lowers the control order to a direct type of control **
-D NO_WRAP ** Turns off wrapping **
-D DEBUG ** Sounds Effects/Printing messages on soundless linux **

-- Full command:
eval "$(cat DE_Minimal.c | grep -m 4 "\-\-cflags cairo")"; cp *.so ../gym-master/gym/envs/space_fortress/linux2

********************************* -------------------------------------- **********************************/

#ifndef DE_H
#define DE_H
#include <math.h>
//#include <cairo.h>
//#if defined(GUI) && defined(__APPLE__)
//	#include <cairo-quartz.h> // Is this available on linux? No!
//#endif
//#include <gtk/gtk.h>
//#include <gdk/gdkkeysyms.h>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#ifndef GLOBALS
#define GLOBALS
#include "myconst.h"
#include "myext.h"
#endif

#include "DE_Minimal.h"
#include "HM.h"
#include "RS.h"

#define RENDER_WIDTH 84
#define RENDER_HEIGHT 84

// calculated from old version
#ifdef GUI_INTERFACE
#define DEFAULT_LINE_WIDTH 3.8
#else
#define DEFAULT_LINE_WIDTH 7.0
#endif

//
float symbols[2] = {};
float* get_symbols()
{
	printf("GETTING SYMBOLS:");
	symbols[0] = 1.4f;	
	symbols[1] = 2.3f;
	//for (int i = 3; i >= 0; i--)
	//	printf("%f, ",symbols[i]);
	//printf("\n");
	return symbols;

}

void Initialize_Graphics(cairo_t *cr)
{
//	int Height,OldMaxX;
//	int t,t1; // t is unused
//	int t1;
	int x,dx;

	MaxX = WINDOW_WIDTH;
	MaxY = WINDOW_WIDTH;

	#ifdef GUI_INTERFACE
	cairo_scale(SF_rgb_context, 1, 1);
	#endif

	#ifdef GUI
	cairo_scale(cr, 1, 1);
	#else
	cairo_scale(cr, 1.0/SCALE_F, 1.0/SCALE_F);
	#endif

	cairo_set_antialias(cr, CAIRO_ANTIALIAS_BEST);
	cairo_set_line_cap(cr, CAIRO_LINE_CAP_SQUARE);
	cairo_set_line_join(cr, CAIRO_LINE_JOIN_MITER);

	cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);

//	cairo_set_line_width(cr, 10); // Line width equal to one pixel
	if(cairo_surface_get_type(cairo_get_target(cr)) == CAIRO_SURFACE_TYPE_XLIB)
	{
		// Supply a value VAL between 100.0 and 240.0 (as a double)
		cairo_set_line_width(cr, (135.0 * 1) / ((double) MaxY * 1));
	}
	else if(cairo_surface_get_type(cairo_get_target(cr)) == CAIRO_SURFACE_TYPE_IMAGE)
	{
		#ifdef __APPLE__
			cairo_set_line_width(cr, 8.2);
		#else
			cairo_set_line_width(cr, DEFAULT_LINE_WIDTH);
		#endif
	}
	else // Mostly quartz?
	{
		cairo_set_line_width(cr, (95.1 * 1) / ((double) MaxY * 1)); // for image_surf use 239
	}
//	cairo_set_line_width(cr, (90.1 * 1) / ((double) MaxY * 1));


	dx=MaxX/8;
	Points_X=x=2*TEXT_WIDTH;
	x=x+dx; Control_X=x;
	x=x+dx; Velocity_X=x;
	x=x+dx; Vulner_X=x;
	x=x+dx; IFF_X=x;
	x=x+dx; Interval_X=x;
	x=x+dx; Speed_X=x;
	x=x+dx; Shots_X=x;
}

void Close_Graphics(cairo_t *cr)
{
  cairo_destroy(cr);
}

void Close_Graphics_SF()
{
  cairo_surface_destroy(surface);
  Close_Graphics(SF_canvas);
}


float Fcos(int Headings_Degs) /* compute cos of 0 - 359 degrees */
{
		float arc;
		arc=Headings_Degs*ARC_CONV;
		return(cos(arc));
}

float Fsin(int Headings_Degs) /* compute sin of 0 - 359 degrees */
{
	float arc;
	arc=Headings_Degs*ARC_CONV; /* convert degrees to radians */
	return(sin(arc));
}

void cairo_line(cairo_t *cr, int x1, int y1, int x2, int y2)
{
//	snapCoords(canvas, &x1, &y1 );
//

// This code generates straighter and sharper lines, but also drops parts of objects for
// some reason //	cairo_user_to_device(cr, &x1, &y1);
//	x1 = round(x1) + 0.5;
//	y1 = round(y1) + 0.5;
//	cairo_device_to_user(cr, &x1, &y1);
// //	cairo_user_to_device(cr, &x2, &y2);
//	x2 = round(x2) + 0.5;
//	y2 = round(y2) + 0.5;
//	cairo_device_to_user(cr, &x2, &y2);
	cairo_move_to(cr, x1, y1);
	cairo_line_to(cr, x2, y2);
//	cairo_move_to(cr, x1, y1);
//	cairo_line_to(cr, x2, y2);

}


// Clip within the bounding box of the current_path()
void clip_path_rect(cairo_t *cr)
{
	double x1;
	double y1;
	double x2;
	double y2;
	cairo_path_extents(cr,&x1,&y1,&x2,&y2);
	cairo_path_t *ol_path = cairo_copy_path(cr);
	cairo_new_path(cr);
	// Create the bounding box
	cairo_rectangle(cr, x1-1, y1-1, (x2-x1)+1, (y2-y1)+1);
//	cairo_set_source_rgba(cr, 1, 0, 0, 1);
//	cairo_stroke_preserve(cr);
	cairo_clip(cr);
	// Restore the old path

	cairo_append_path(cr,ol_path);
}

void stroke_in_clip(cairo_t *cr)
{
	clip_path_rect(cr);
	cairo_stroke(cr);
	cairo_reset_clip(cr);
}

void set_initial_vals()
{
	Loop_Counter = 0;
	intv_t1 = 0;
	intv_t2 = 0;
//	Terminal_State = 0;
	Init_Aim_Session();

//	cairo_path_t *empty_path = cairo_copy_path(cr);
//	PrevShip = empty_path;
//	for(int i = 0; i < MAX_NO_OF_MISSILES; i++)
//	{
//		PrevMissile[i] = empty_path;
//	}

//	Set_Bonus_Chars();	// Probably not needed because we don't need to save them in memory
	// first or whatver
//	Points_Should_Update = 1;
//	Velocity_Should_Update = 1;
//	Speed_Should_Update = 1;
//	Vulner_Should_Update = 1;
//	Interval_Should_Update = 1;
//	Shots_Should_Update = 1;
//	Control_Should_Update = 1;

//	PrevFort = empty_path;
//	PrevMine = empty_path;
//	PrevShell = empty_path;
//	memset(Missile_Should_Update, 0, MAX_NO_OF_MISSILES);
//	memset(Missile_Should_Clean, 0, MAX_NO_OF_MISSILES);
	Reset_Screen();

}

// Returns the screen as rendered by the minimal game renderer
unsigned char* get_screen()
{
	return cairo_image_surface_get_data(surface);
}

// Returns a pointer to an image canvas with therein a full sized, full color render of the
// game
unsigned char* get_original_screen()
{
	#ifdef GUI_INTERFACE
	return cairo_image_surface_get_data(rgb_surface);
	#endif
	return 0;
}


unsigned char* update_screen()
{
	#ifdef GUI_INTERFACE
	clean(SF_rgb_context);
	#endif
	#ifdef GUI_INTERFACE
	update_drawing(SF_rgb_context);
	#endif
	clean(SF_canvas);
	update_drawing(SF_canvas);
	return cairo_image_surface_get_data(surface);
}

void set_key(int key_value)
{
	Lastkey = Key;
	Key = key_value;
	New_Input_Flag=ON;

}

// Placed here to center the whole interface in one file
// (which might eliminate the GTK support)
// For the python interface
float get_score()
{
	float reward = Score;
	Score = 0.0;
	return reward;
}

// Resets the Space fortress game (i.e. the non gtk standard drawing for learning surface)
void reset_sf()
{
	Initialized_Graphics = 0;
	set_initial_vals();
//	Reset_Screen(SF_canvas);
}

int get_terminal_state()
{
	if(Terminal_State)
	{
		Terminal_State = 0;
		return 1;
	}
	else
	{
		return 0;
	}
}



// Cleans all the previous paths from the context for the objects in need of an update
void clean(cairo_t *cr)
{
	#ifdef GUI
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	#else
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.0);
	#endif

	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 0.0, 0.0, 0.0);
	#endif
	cairo_paint(cr);
}



// Draws the hexagon around the fortress, maybe drop this?
void Draw_Hexagone(cairo_t *cr,int X_Center,int Y_Center,int Hex_Outer_Radius)
{
	int Abs_Y;
//	int svcolor;

//	svcolor=getcolor(); /* save present color */
//	setcolor(HEX_COLOR);
	cairo_set_source_rgba(cr, SF_GREEN);
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 0.0, 0.66, 0.0);
	#endif
	Abs_Y=Hex_Outer_Radius*0.866;	/* sin(60)=0.866 */
	cairo_move_to(cr, X_Center+Hex_Outer_Radius,Y_Center); /* right-hand tip */
	cairo_line_to(cr, X_Center+Hex_Outer_Radius/2,Y_Center-Abs_Y);
	cairo_line_to(cr, X_Center-Hex_Outer_Radius/2,Y_Center-Abs_Y);
	cairo_line_to(cr, X_Center-Hex_Outer_Radius,Y_Center);
	cairo_line_to(cr, X_Center-Hex_Outer_Radius/2,Y_Center+Abs_Y);
	cairo_line_to(cr, X_Center+Hex_Outer_Radius/2,Y_Center+Abs_Y);
	cairo_line_to(cr, X_Center+Hex_Outer_Radius,Y_Center);

//	cairo_line(cr, X_Center+Hex_Outer_Radius,Y_Center, X_Center+Hex_Outer_Radius/2,Y_Center-Abs_Y);
//	cairo_line(cr, X_Center-Hex_Outer_Radius/2,Y_Center-Abs_Y, X_Center-Hex_Outer_Radius,Y_Center);
//	cairo_line(cr, X_Center-Hex_Outer_Radius/2,Y_Center+Abs_Y, X_Center+Hex_Outer_Radius/2,Y_Center+Abs_Y);
//	cairo_line_to(cr, X_Center+Hex_Outer_Radius,Y_Center);

//	setcolor(svcolor); /* restore previous color */
//	return(0);
}


void Draw_Ship_Nose(cairo_t *cr, int x, int y, int Headings, int size)
{

	int x1,y1;	/* ship's aft location */
	int x2,y2;	/* ship's nose location */

	#ifdef GUI
	cairo_set_source_rgb(cr, 0.0, 0.66, 0.0);
	#else
	cairo_set_source_rgba(cr, SF_GREEN);
	#endif


	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 0.0, 0.66, 0.0);
	#endif

	x1=x-0.5*size*Fsin(Headings);
	y1=y+0.5*size*Fcos(Headings);
	x2=x+0.5*size*Fsin(Headings);
	y2=y-0.5*size*Fcos(Headings);
	cairo_line(cr,x1,y1,x2,y2);

}

void Draw_Ship(cairo_t *cr, int x, int y, int Headings, int size)
{
	/* size - is the entire length of the ship */
	int xl,yl;	 /* ship's left wing tip location */
	int xr,yr;	 /* ship's right wing tip location */
	int xc,yc;	/* fuselage and wings connecting point */
	int Right_Wing_Headings;
	int Left_Wing_Headings;
//	int svcolor;
//	float tmp;  // Unused

//	svcolor=getcolor(); /* save present color */
//	setcolor(SHIP_COLOR); // yellow
	cairo_set_source_rgba(cr, SF_YELLOW);
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 1.0, 1.0, 0.33);
	#endif
	xc=x;
	yc=y;
	Right_Wing_Headings=Headings+135;
	if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
	Left_Wing_Headings=Headings+225;
	if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
	xr=xc+0.707*size*Fsin(Right_Wing_Headings);
	yr=yc-0.707*size*Fcos(Right_Wing_Headings);
	cairo_line(cr,xc,yc,xr,yr);
	xl=xc+0.707*size*Fsin(Left_Wing_Headings);
	yl=yc-0.707*size*Fcos(Left_Wing_Headings);
	cairo_line(cr,xc,yc,xl,yl);
//	PrevShip = cairo_copy_path(cr);
//	setcolor(svcolor); /* restore previous color */
//	return(0);
}


void Draw_Mine(cairo_t *cr, int x, int y, int size)	/* x,y is on screen center location
					size is half of horizontal diagonal */
{
//	int svcolor;

//	svcolor=getcolor(); /* save present color */
//	setcolor(MINE_COLOR); // maybe different than blue for easier recogniztion?
	cairo_set_source_rgba(cr, SF_BLUE);
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 0.33, 1.0, 1.0);
	#endif
	cairo_move_to(cr,x-size,y);
	cairo_line_to(cr,x,y-1.18*  size);	 /* 1.3/1.1=1.18 */
	cairo_line_to(cr,x+size,y);
	cairo_line_to(cr,x,y+1.18* size);
	cairo_line_to(cr,x-size,y);
//	cairo_line(cr,x-size,y,x,y-1.18*size);
//	cairo_line(cr,x,y-1.18*size,x+size,y);
//	cairo_line(cr,x+size,y,x,y+1.18*size);
//	cairo_line(cr,x,y+1.18*size,x-size,y);
//	PrevMine = cairo_copy_path(cr);
//	setcolor(svcolor); /* restore previous color */
}

void Draw_Missile (cairo_t *cr, int x, int y, int Headings, int size, int missile_idx)
{
	int x1,y1;	/* ship's aft location */
	int x2,y2;	/* ship's nose location */
	int xl,yl;	 /* ship's left wing tip location */
	int xr,yr;	 /* ship's right wing tip location */
	int xc,yc;	/* fuselage and wings	 connecting point */
	int Right_Wing_Headings;
	int Left_Wing_Headings;
//	int svcolor;

//	svcolor=getcolor(); /* save present color */
//	setcolor(MISSILE_COLOR);
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.2);
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 1.0, 0.0, 0.0);
	#endif
	x1=x;
	y1=y;
	x2=x1+size*Fsin(Headings);
	y2=y1-size*Fcos(Headings);
	cairo_line(cr, x1,y1,x2,y2);
	xc=x2;
	yc=y2;
	Right_Wing_Headings=Headings+135;
	if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
	Left_Wing_Headings=Headings+225;
	if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
	xr=xc+0.25*size*Fsin(Right_Wing_Headings);
	yr=yc-0.25*size*Fcos(Right_Wing_Headings);
	cairo_line(cr,xc,yc,xr,yr);
	xl=xc+0.25*size*Fsin(Left_Wing_Headings);
	yl=yc-0.25*size*Fcos(Left_Wing_Headings);
	cairo_line(cr,xc,yc,xl,yl);
//	PrevMissile[missile_idx] = cairo_copy_path(cr);
//	setcolor(svcolor); /* restore previous color */
}

void update_drawing(cairo_t *cr)
{
	// Effect Flag is first time around animation iteration
//	if ((!Jitter_Flag && !Explosion_Flag) || Effect_Flag)
//	{
	Draw_Ship(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);
	cairo_set_line_width(cr, DEFAULT_LINE_WIDTH+1.5);
	Draw_Ship_Nose(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);
	cairo_set_line_width(cr, DEFAULT_LINE_WIDTH);

//		cairo_bounding_box(cr);
//		Ship_Should_Update = 0;
//	}

//	Draw_Fort(cr, MaxX/2,MaxY/2,Fort_Headings,FORT_SIZE_FACTOR*MaxX);
//	stroke_in_clip(cr);
//		Fort_Should_Update = 0;
	for(int i=0;i<MAX_NO_OF_MISSILES;i++)
	{
		if (Missile_Flag[i]==ALIVE)
		{
			Draw_Missile(cr, Missile_X_Pos[i], Missile_Y_Pos[i], Missile_Headings[i], MISSILE_SIZE_FACTOR*MaxX, i);
			cairo_stroke(cr);
		}
	}
	if (Mine_Flag==ALIVE)
	{
		Draw_Mine(cr, Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
		cairo_stroke(cr);
	}

}

float Find_Headings(int x1, int y1, int x2, int y2)
{
//	int quadrant;	// Unused
	double arcsinalfa;
	double b;
	double a;
	arcsinalfa=abs(x1-x2);
	a=pow(x1-x2,2)+pow(y1-y2,2);
	b=sqrt(a);
	arcsinalfa=asin(arcsinalfa/b);
	if (x1<x2)
		if (y1>y2) /* quadrant=1 */ return(arcsinalfa*57.3+0.5);
	else
		/* quadrant=2 */ return(180.0-arcsinalfa*57.3+0.5);
	 else
	 if (y1>y2) /* quadrant=4 */ return(360.0-arcsinalfa*57.3+0.5);
	else
		/* quadrant=3 */ return(180.0+arcsinalfa*57.3+0.5);
}


void start_drawing()
{
//	int win_width = (int) (((float) WINDOW_WIDTH)/(float)SCALE_F);
//	int win_height = (int) (((float) WINDOW_HEIGHT)/(float)SCALE_F);
	surface = cairo_image_surface_create(CAIRO_FORMAT_A8, RENDER_WIDTH, RENDER_HEIGHT);
	#ifdef GUI_INTERFACE
	rgb_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB16_565, WINDOW_WIDTH, WINDOW_HEIGHT);
	SF_rgb_context = cairo_create(rgb_surface);
	#endif
	SF_canvas = cairo_create(surface);
	Initialize_Graphics(SF_canvas);
	reset_sf();
	// restore the line width
//	cairo_set_line_width(SF_canvas, (224.1 * 1) / ((double) MaxY * 1));
//	Draw_Frame(SF_canvas); // Draw the basis
	// Done in reset_sf -> set_initial_vals -> reset_screen now.
}

void stop_drawing()
{
	// Specific SF_canvas function
	Close_Graphics_SF();
	#ifdef GUI_INTERFACE
  Close_Graphics(SF_rgb_context);
	cairo_surface_destroy(rgb_surface);
	#endif
}

unsigned char* update_frame()
{
	// This should have the form clean -> sf_iter -> update, because bottom panel text will in
  // this  way be ereased, numerically updated, and then visually updated
	clean(SF_canvas);
	#ifdef GUI_INTERFACE
	clean(SF_rgb_context);
	#endif
	SF_iteration();
	update_drawing(SF_canvas);
	#ifdef GUI_INTERFACE
	update_drawing(SF_rgb_context);
	#endif

//	cairo_line(SF_canvas, 0, MaxY, MaxX, MaxY );
//	cairo_stroke(SF_canvas);
//	cairo_surface_t *surface2 = cairo_image_surface_create(CAIRO_FORMAT_A8, WINDOW_WIDTH/SCALE_F, WINDOW_HEIGHT/SCALE_F);

//	printf("Hey [1]\n");
//	cairo_t *des = cairo_create (surface2);
//	cairo_set_operator(des, CAIRO_OPERATOR_HSL_LUMINOSITY);
//	cairo_set_source_surface (des, surface, 0, 0);
//	printf("Hey [2]\n");
//	cairo_paint(des);
//	printf("Hey [3]\n");
	return cairo_image_surface_get_data(surface);
}

// Converts degrees to radians
float deg2rad(int deg)
{
	return deg * (M_PI / 180.0);
}


void Reset_Screen()
{
        /*  reset variables */
//		printf("Maxxes: %d %d \n", MaxX, MaxY);
    Ship_X_Pos=0.25*MaxX; /* on a 640 x 480 screen VGA-HI */
    Ship_Y_Pos=0.5*MaxY; /* same as above */
//		printf("Ship pos after reset: %d %d\n", Ship_X_Pos, Ship_Y_Pos);

    Ship_Headings=0;
    Mine_Flag=DEAD;
    for(int i=0;i<MAX_NO_OF_MISSILES;i++) Missile_Flag[i]=DEAD;
    Missile_Type=VS_FRIEND;
    Missile_Vs_Mine_Only=OFF;
    Missiles_Counter=0;
    Shell_Flag=DEAD;
    Rotate_Input=0; /* joystick left/right */
    Accel_Input=0; /* joystick forward */
    End_Flag=OFF;
    // Fort_Headings=270;
    Vulner_Counter=0;
    Timing_Flag=OFF; /* if screen reset between consecutive presses */
    Resource_Flag=OFF;
    Resource_Off_Counter=0;
    Bonus_Display_Flag=NOT_PRESENT;   /* in case bonus is pressed after */
    Bonus_Granted=OFF;

	Score=0.0;

        /*  reset variables */
    Ship_X_Pos=0.5*MaxX; /* on a 640 x 480 screen VGA-HI */
    Ship_Y_Pos=0.5*MaxY; /* same as above */
    Ship_X_Speed=0.0;
    Ship_Y_Speed=0.0;
    Ship_Headings=0;
    Mine_Flag=DEAD;
    for(int i=1;i<6;i++) Missile_Flag[i]=DEAD;
    Missile_Type=VS_FRIEND;
    Missile_Vs_Mine_Only=OFF;
    Missiles_Counter=0;


}  /* end reset screen */


#endif
