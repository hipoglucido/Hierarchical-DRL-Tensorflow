// Ubuntu upstream clang is not at 3.9 yet, so alias clang to clang3.9 on Ubuntu
/****************************************** COMPILE THINGY **************************************

|** Compile a shared library **|
# You can copy/paste all of this 😘
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3 $Switches;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o control_frame_lib.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 $Switches;
# *** Add -D GUI_INTERFACE to enable acces to full size, full color renders of the game *** #
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3 -D GUI_INTERFACE $Switches;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o control_frame_lib_FULL.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 -D GUI_INTERFACE $Switches;

|** Compile a playable GUI version **
clang -Wall -g -fPIC myvars.c TCOL.c DE_Minimal.c HM.c RS.c `pkg-config --cflags cairo pkg-config --libs cairo pkg-config --cflags gtk+-3.0 pkg-config --libs gtk+-3.0 ` -lm -o Control -Wno-dangling-else -Wno-switch -D GUI $Switches

-- Switches:
-D GUI_INTERFACE ** Full sized and colored game renders **
-D GRID_MOVEMENT ** Lowers the control order to a direct type of control **
-D NO_WRAP ** Turns off wrapping **
-D NO_DIRECTION ** Turns off movement based on the ships nose direction **
-D DEBUG ** Sounds Effects/Printing messages on soundless linux **
-D ROTATE_ANGLE=theta ** The rotation of the ship in degrees **
-D NO_RANDOM_SPAWN ** Disable random spawn location and random ship orientation **

-- Full command:
eval "$(cat DE_Minimal.c | grep -m 4 "\-\-cflags cairo")"; cp *.so ../gym-master/gym/envs/space_fortress/linux2

***************************** -------------------------------------- ******************************/


#ifndef DE_H
#define DE_H
#include <math.h>

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

#ifdef CV_SCALE
//#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#endif

#include "DE_Minimal.h"
#include "HM.h"
#include "RS.h"

#define RENDER_WIDTH 84
#define RENDER_HEIGHT 84


#ifdef GUI_INTERFACE
#define DEFAULT_LINE_WIDTH 2.8
#else
#define DEFAULT_LINE_WIDTH 7.0
#endif

// Globals
#ifdef CV_SCALE
CvSize size;
CvSize outsize;
IplImage *frame;
IplImage *outframe;
#endif

float symbols[6] = {};
int get_screen_width(){
	return WINDOW_WIDTH;
}
int get_screen_height(){
	return WINDOW_HEIGHT;
}
float* get_symbols()
{
	//printf("GETTING SYMBOLS:");
	symbols[0] = Ship_Y_Pos;// /(float) WINDOW_WIDTH;	
	symbols[1] = Ship_X_Pos;// /(float) WINDOW_HEIGHT;
	symbols[2] = Ship_Headings;// /(float) 360;
	symbols[3] = Square_Y;// /(float) WINDOW_WIDTH;
	symbols[4] = Square_X;// /(float) WINDOW_HEIGHT;
	symbols[5] = Square_Step;// /(float) MAX_SQUARE_STEPS;
	//for (int i = 3; i >= 0; i--)
	//	printf("%f, ",symbols[i]);
	//printf("\n");
	return symbols;
}

int is_frictionless(){
	#ifdef GRID_MOVEMENT
		return 0;
	#else
		return 1;
	#endif
}

int is_wrapper(){
	#ifdef NO_WRAP
		return 0;
	#else
		return 1;
	#endif
}

int is_no_direction(){
	#ifdef NO_DIRECTION
		return 1;
	#else
		return 0;
	#endif
}

void Initialize_Graphics(cairo_t *cr)
{
//	int Height,OldMaxX;
//	int t,t1; // t is unused
//	int t1;
	int x,dx;

	MaxX = WINDOW_WIDTH;
	MaxY = WINDOW_WIDTH;

	#ifdef CV_SCALE
	size = cvSize(RENDER_WIDTH, RENDER_HEIGHT);
	outsize = cvSize(SCALE_WIDTH, SCALE_HEIGHT);
	frame = cvCreateImage(size, IPL_DEPTH_8U, 1);
	outframe = cvCreateImage(outsize, frame->depth, frame->nChannels);
	#endif

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

	if(cairo_surface_get_type(cairo_get_target(cr)) == CAIRO_SURFACE_TYPE_XLIB)
	{
		// Supply a value VAL between 100.0 and 240.0 (as a double)
		cairo_set_line_width(cr, (435.0 * 1) / ((double) MaxY * 1));
	}
	else if(cairo_surface_get_type(cairo_get_target(cr)) == CAIRO_SURFACE_TYPE_IMAGE)
	{
		#ifdef __APPLE__
			cairo_set_line_width(cr, DEFAULT_LINE_WIDTH);
		#else
			cairo_set_line_width(cr, DEFAULT_LINE_WIDTH);
		#endif
	}
	else // Mostly quartz?
	{
		cairo_set_line_width(cr, (390.1 * 1) / ((double) MaxY * 1)); // for image_surf use 239
	}
	// We need the options to turn off font anti-aliasing
	cairo_font_options_t *font_options = cairo_font_options_create();
	cairo_set_font_size(cr, POINTS_FONT_SIZE);

	// Turning off anti-alaising
	cairo_get_font_options(cr, font_options);
	cairo_font_options_set_antialias(font_options, CAIRO_ANTIALIAS_BEST);
	cairo_set_font_options(cr, font_options);
	cairo_select_font_face(cr,"DriodSans",CAIRO_FONT_SLANT_NORMAL,CAIRO_FONT_WEIGHT_NORMAL);
	cairo_font_options_destroy(font_options);

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

	cairo_move_to(cr, x1, y1);
	cairo_line_to(cr, x2, y2);

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
//	cairo_set_source_rgba(canvas, 1, 0, 0);
//	cairo_stroke_preserve(canvas);
	cairo_clip(cr);
	// Restore the old path

	cairo_append_path(cr,ol_path);
}

void stroke_in_clip(cairo_t *cr) // Uncomment stuff maybe
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
	t0 = 0;
	Reset_Screen();

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


void set_key(int key_value)
{
	Lastkey = Key;
	Key = key_value;
	if (key_value != WAIT)
	{
		New_Input_Flag=ON;
	}
	//printf("KEEEEEEEEEEY %d\n", key_value);
}

// For the python interface:
// Does not actually return the score anymore, but returns a reward
float get_score()
{
	float reward = Score;
	Score = 0.0;
	//printf("Reward: %f\n", reward);
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

// Apperently normal stroking is faster than stroking in a clipping area
void update_drawing(cairo_t *cr)
{

	cairo_set_line_width(cr, DEFAULT_LINE_WIDTH);
	Draw_Ship(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);
	cairo_set_line_width(cr, DEFAULT_LINE_WIDTH+1.5);
	Draw_Ship_Nose(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);

	Draw_Square(cr, Square_X, Square_Y);
	cairo_stroke_preserve(cr);
	cairo_fill(cr);
}

void Draw_Square(cairo_t *cr, int x, int y)
{
	#ifdef GUI
	cairo_set_source_rgb(cr, 1.0, 0.33, 0.33);
	#else
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.72);
	#endif


	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 1.0, 0.66, 0.66);
	#endif
	
	
	cairo_rectangle (cr, x, y, SQUARE_WIDTH, SQUARE_HEIGHT);
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

	#ifdef GUI
	cairo_set_source_rgb(cr, 1.0, 1.0, 0.33);
	#else
	cairo_set_source_rgba(cr, SF_YELLOW);
	#endif

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
	#ifdef CV_SCALE
	cvSetData(frame, cairo_image_surface_get_data(surface), RENDER_WIDTH);
	// Works til resize, saving the im also works
	cvResize(frame, outframe, CV_INTER_AREA);
//	Mat m;
//	iplImageToMat(frame, 1);
//	printf("SCALING YEAH! %d %d\n", outframe.width, outframe.height);
	// cvCvtColor(frame,outframe,CV_RGB2XYZ);
	return (unsigned char *)outframe->imageData;
	#endif
	return cairo_image_surface_get_data(surface);
}

//// Converts degrees to radians
//float deg2rad(int deg)
//{
//	return deg * (M_PI / 180.0);
//}


void Reset_Screen()
{
        /*  reset variables */
	#ifdef NO_RANDOM_SPAWN
    Ship_X_Pos=0.5*MaxX;
    Ship_Y_Pos=0.5*MaxY;
    Ship_Headings=0;
    #else
    Ship_X_Pos=randrange(20,MaxX-20);
    Ship_Y_Pos=randrange(20,MaxY-20);
    Ship_Headings=randrange(0,359);
    #endif
//		printf("Ship pos after reset: %d %d\n", Ship_X_Pos, Ship_Y_Pos);
    Ship_X_Speed=0.0;
    Ship_Y_Speed=0.0;
    Rotate_Input=0; /* joystick left/right */
    Accel_Input=0; /* joystick forward */

	Score=0.0;

//	srand(time(NULL));
	N_Squares = 0;
	Square_Step = MAX_SQUARE_STEPS;
	Square_Flag = KILL;
	Prev_Ship_Square_Dist = 0;
	Prev_Ship_Square_Dist_Ratio = 1;
	Square_X = 0;
	Square_Y = 0;
}  /* end reset screen */


#endif
