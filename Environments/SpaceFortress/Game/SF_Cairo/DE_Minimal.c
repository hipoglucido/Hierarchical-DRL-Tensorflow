/****************************************** COMPILE THINGY **************************************

|** Compile a shared library **|
# You can copy/paste all of this 😘
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3 $Switches;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o sf_frame_lib.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 $Switches;
# *** Add -D GUI_INTERFACE to enable acces to full size, full color renders of the game *** #
clang -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c  myvars.c DE_Minimal.c HM.c TCOL.c RS.c -Wno-dangling-else -Wno-switch -O3 -D GUI_INTERFACE $Switches;
clang -march=native `pkg-config --cflags cairo --libs cairo` -shared -o sf_frame_lib_FULL.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 -D GUI_INTERFACE $Switches;

|** Compile a playable GUI version **|
clang -Wall -g -fPIC myvars.c TCOL.c DE_Minimal.c HM.c RS.c `pkg-config --cflags cairo pkg-config --libs cairo pkg-config --cflags gtk+-3.0 pkg-config --libs gtk+-3.0 ` -lm -o SpaceFortress -Wno-dangling-else -Wno-switch -D GUI $Switches

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

#include "DE_Minimal.h"
#include "HM.h"
#include "RS.h"

int jitter_switch = 1;


#define RENDER_WIDTH 84
#define RENDER_HEIGHT 84


#ifdef GUI_INTERFACE
#define DEFAULT_LINE_WIDTH 3.8
#else
#define DEFAULT_LINE_WIDTH 7.0
#endif


////////////////////////////////////
float symbols[11] = {};
int get_screen_width(){
	return WINDOW_WIDTH;
}
int get_screen_height(){
	return WINDOW_HEIGHT;
}
float* get_symbols()
{
	//printf("GETTING SYMBOLS:");
	symbols[0]  = Ship_Y_Pos;// /(float) WINDOW_WIDTH;	
	symbols[1]  = Ship_X_Pos;// /(float) WINDOW_HEIGHT;
	symbols[2]  = Ship_Y_Speed;
	symbols[3]  = Ship_X_Speed;
	symbols[4]  = Ship_Headings;// /(float) 360;
	symbols[5]  = Shell_Y_Pos;//(float) Missile_Y_Pos;
	symbols[6]  = Shell_X_Pos;//(float) Missile_X_Pos;	
	symbols[7]  = Fort_Headings;
	symbols[8]  = Missile_Stock;
	symbols[9]  = Mine_Y_Pos;
	symbols[10] = Mine_X_Pos;
	//shell speed?
	//for (int i = 3; i >= 0; i--)
	//printf("SPPED Y %f, ",symbols[6]);
	// printf("X %f, ",(double) Mine_X_Pos);
	// printf("Y %f, ",(double) Mine_Y_Pos);
	//printf("\n");
	return symbols;
}


int did_I_hit_mine(){
	int result = Mine_Hit;
	Mine_Hit  = 0;
	return result;
}
int did_I_hit_fortress(){
	int result = Fortress_Hit;
	Fortress_Hit = 0;
	return result;
}

int did_mine_hit_me(){
	int result = Hit_by_Mine;
	Hit_by_Mine  = 0;
	return result;
}
int did_fortress_hit_me(){
	int result = Hit_by_Fortress;
	Hit_by_Fortress = 0;
	return result;
}


int get_vulner_counter(){
	return Vulner_Counter;
}
int get_lifes_remaining(){
	return SHIP_LIFES - Ship_Killings_Counter;
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

////////////////////////////////////


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
//	cairo_set_line_width(cr, (90.1 * 1) / ((double) MaxY * 1));

////	 Cairo uses a different coordinate system than graphics.h, so we reflect Cairo's through
////	 the x-asis to make it equal to that of graphics.h.
////	cairo_matrix_t x_reflection_matrix;
	// Reflecting it however means that text will also be reflected. We therefore also use a
	// reflection matrix for drawing fonts to reflect text back.
//	cairo_matrix_t font_reflection_matrix;

	// We need the options to turn off font anti-aliasing
	cairo_font_options_t *font_options = cairo_font_options_create();
//	cairo_matrix_init_identity(&x_reflection_matrix);
//	x_reflection_matrix.yy = -1.0;
//	cairo_set_matrix(cr, &x_reflection_matrix);

	cairo_set_font_size(cr, POINTS_FONT_SIZE);
//	cairo_set_font_size(cr, 5.9);
//	cairo_get_font_matrix(cr, &font_reflection_matrix);
//	font_reflection_matrix.yy = font_reflection_matrix.yy * -1;
//	font_reflection_matrix.x0 += side_panel_size; // see (1)

//	cairo_set_font_matrix(cr, &font_reflection_matrix);
	// Translate negative height down because the reflection draws on top of the drawing surface
	// (i.e. out of frame, directly on top of the frame)

	// (1) Also translate the matrix over the x-axis to emulate the fact that DOS places the SF
	// square in the middle.
//	cairo_translate(cr, side_panel_size, -MaxY);
//	cairo_translate(cr, 0, -MaxY);

	// Turning off anti-alaising
	cairo_get_font_options(cr, font_options);
	cairo_font_options_set_antialias(font_options, CAIRO_ANTIALIAS_BEST);
	cairo_set_font_options(cr, font_options);
	cairo_select_font_face(cr,"DriodSans",CAIRO_FONT_SLANT_NORMAL,CAIRO_FONT_WEIGHT_NORMAL);
	cairo_font_options_destroy(font_options);


	// Sets all the values in the array to the empty path
	// Gives a vague warning, probably because this only works for types with the size of an int
	// (source: SO)
//	memset(PrevMissile, empty_path, MAX_NO_OF_MISSILES);
//	PrevMissile = {empty_path};
	// Attemps above don't work, so we initialize manually

	// clears the screen, probably the dos screen, and sets the current graphics write
	// pointer to (0,0)
	//	cleardevice();

	//	The "textheight" function returns the height of a string in pixels.
//	Height=textheight("H"); /* Get basic text height */
//	Height = TEXT_HEIGHT;
//
// 	OldMaxX=MaxX;
//  t1=4*Height;
//
//  Panel_Y_End=MaxY;
//  Panel_Y_Start=MaxY-t1+2;
//  MaxY_Panel=Panel_Y_End-Panel_Y_Start;
//  MaxY=MaxY-t1;
//	MaxX=MaxY;

	// Any modern graphics library should handle this "if" statements themselves, if needed at
	// all because we don't really need to know anymore wether or not we are on a vga display.
	// This aspect ratio stuff has to do with the fact if your display has square pixels or not.
	// AspectRatio is defined in opengraphics. Pixels are always square nowadays
//	if(AspectRatio==1.0) /* VGA HI */ square pixel ratio
//	MaxX=MaxY;
//	else	/* for all others */
//	{
//		MaxX=MaxX*AspectRatio;	/********* MaxX and MaxY give a square */
//		MaxX=MaxX-t1/AspectRatio;	/******** less two panel lines */
//	}
//	Xmargin=OldMaxX/2-MaxX/2;
//	printf("Xmargin: %d", Xmargin);
//	cairo_translate(cr, Xmargin, 0);
	// -- void setviewport(int left, int top, int right, int bottom, int clip);
	// setviewport function is used to restrict drawing to a particular portion on the screen. 	// For example "setviewport(100 , 100, 200, 200, 1)" will restrict our drawing activity
	// inside the rectangle(100,100, 200, 200).
	//
	// left, top, right, bottom are the coordinates of main diagonal of rectangle in which we wish to restrict our drawing. Also note that the point (left, top) becomes the new origin.
//	setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);

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
	t0 = 0;
	jitter_switch = 1;
//	Terminal_State = 0;
	Init_Game();
	//Select_Mine_Menus();

//	cairo_path_t *empty_path = cairo_copy_path(cr);
//	PrevShip = empty_path;
//	for(int i = 0; i < MAX_NO_OF_MISSILES; i++)
//	{
//		PrevMissile[i] = empty_path;
//	}
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
	New_Input_Flag=ON;
//  A list of GTK hex key values as decimals
//	GDK_KEY_1 0xffbe 65470
//	GDK_KEY_2 0xffbf 65471
//	GDK_KEY_3 0xffc0 65472
//	GDK_KEY_Left 0xff51 65361
//	GDK_KEY_Up 0xff52 65362
//	GDK_KEY_space 0x020 32
}

// For the python interface:
// Does not actually return the score anymore, but returns a reward
float get_score()
{
	float reward = Score;
	Score = 0.0;
	Penalize_Wrap = 0;
	//Mine_Hit = 0;
	//Fortress_Hit = 0;
	return reward;
}

// Resets the Space fortress game (i.e. the non gtk standard drawing for learning surface)
void reset_sf()
{
	#ifdef GUI
	Initialized_Graphics = 0;
	#endif
	set_initial_vals();
//	Reset_Screen(SF_canvas);
}

int get_terminal_state()
{
	if(Terminal_State)
	{
		Terminal_State = 0;
		// return Terminal_State_Flag; // Nice to have one day
		return 1;
	}
	else
	{
		return 0;
	}

}

// W and H are the dimensions of the clipping rectangle
void cairo_clip_text(cairo_t *cr, int x1, int y1, int w,  int h)
{
	cairo_rectangle(cr, x1, y1, w, h); // Not optimal
//	cairo_stroke_preserve(cr);
	cairo_clip(cr);
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


void update_drawing(cairo_t *cr)
{

	if(Explosion_Flag)
	{
		// This actually does nothing the first time around
		// explosion_step2 is sort of the inner, yellow circle, one step behind step1
		for(int i = 0; i < Explosion_Step+1; i++)
		{
			explosion_step2(cr, ExpX, ExpY, i);
		}
		explosion_step1(cr, ExpX, ExpY, Explosion_Step);
	
		if((Explosion_Step * 10) >= ExpRadius)
		{
			Terminal_State = Terminal_State_Flag;
			Explosion_Step = 0;
			Explosion_Flag = 0;
			set_initial_vals();
		}
	}
	else if(Jitter_Flag)
	{
	
		if(jitter_switch)
		{
			jitter_step1(cr, Jitter_Step);
		}
		else
		{
			jitter_step2(cr, Jitter_Step);
		}
	
		if(Jitter_Step < 1)
		{
			Jitter_Step = 8;
			Jitter_Flag = 0;
		}
	}

	Draw_Ship(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);
	Draw_Ship_Nose(cr, Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);

	Draw_Fort(cr, MaxX/2,MaxY/2,Fort_Headings,FORT_SIZE_FACTOR*MaxX);
	cairo_stroke(cr);

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

	if(Shell_Flag==ALIVE)
	{
		Draw_Shell(cr, Shell_X_Pos,Shell_Y_Pos,Shell_Headings,SHELL_SIZE_FACTOR*MaxX);
		cairo_stroke(cr);
	}

//	Update_Points(cr);
//	Update_Velocity(cr);
//	Update_Speed(cr);
//	Update_Vulner(cr);
//	Update_Interval(cr);
//	Update_Shots(cr);
//	Update_Control(cr);

//	if(Mine_Flag==ALIVE)
//	{
//
//		Draw_Mine_Type(cr);
////		Mine_Type_Should_Update = 0;
//	}
//	if(Bonus_Char_Should_Update) // Set to always update (i.e. change nothing)
//	{
//		Draw_Bonus_Char(cr);
//		cairo_reset_clip(cr);
//	}

}

void Draw_Fort(cairo_t *cr, int x, int y, int Headings, int size )
{
	int x1,y1;	 /* fort's aft location */
	int x2,y2;	 /* fort's nose location */
	int xl,yl;	 /* ship's left wing tip location */
	int xr,yr;	 /* ship's right wing tip location */
	int xc,yc;	 /* fuselage and wings connecting point */
	int xrt,yrt;	 /* tip of right wing */
	int xlt,ylt;	 /* tip of left wing */
	int Right_Wing_Headings;
	int Left_Wing_Headings;
	int Right_Wing_Tip_Headings;
	int Left_Wing_Tip_Headings;
//	int svcolor;

//	svcolor=getcolor(); /* save present color */
//	setcolor(FORT_COLOR); // blueee

	#ifdef GUI
	cairo_set_source_rgb(cr, 0.33, 1.0, 1.0);
	#else
	cairo_set_source_rgba(cr, SF_BLUE);
	#endif
	
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 0.33, 1.0, 1.0);
	#endif
	


	x1=x;
	y1=y;
	x2=x1+size*Fsin(Headings);
	y2=y1-size*Fcos(Headings);
	cairo_line(cr,x1,y1,x2,y2);
	xc=x1+(x2-x1)*0.5;
	yc=y1+(y2-y1)*0.5;
	Right_Wing_Headings=Headings+90;
	if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
	Left_Wing_Headings=Headings+270;
	if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
	xr=xc+0.4*size*Fsin(Right_Wing_Headings)+0.5;
	yr=yc-0.4*size*Fcos(Right_Wing_Headings)+0.5;
	cairo_line(cr,xc,yc,xr,yr);
	xl=xc+0.4*size*Fsin(Left_Wing_Headings)+0.5;
	yl=yc-0.4*size*Fcos(Left_Wing_Headings)+0.5;
	cairo_line(cr,xc,yc,xl,yl);
	Right_Wing_Tip_Headings=Right_Wing_Headings+90;
	if(Right_Wing_Tip_Headings>359) Right_Wing_Tip_Headings=
						 Right_Wing_Tip_Headings-360;
	xrt=xr+0.5*size*Fsin(Right_Wing_Tip_Headings)+0.5;
	yrt=yr-0.5*size*Fcos(Right_Wing_Tip_Headings)+0.5;
	cairo_line(cr,xr,yr,xrt,yrt);
	Left_Wing_Tip_Headings=Right_Wing_Tip_Headings;
	xlt=xl+0.5*size*Fsin(Left_Wing_Tip_Headings)+0.5;
	ylt=yl-0.5*size*Fcos(Left_Wing_Tip_Headings)+0.5;
	cairo_line(cr,xl,yl,xlt,ylt);
//	PrevFort = cairo_copy_path(cr);
//	setcolor(svcolor); /* restore previous color */
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

	#ifdef GUI
	cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
	#else
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.2126);
	#endif

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

void Draw_Shell(cairo_t *cr, int x, int y, int Headings, int size)
{
	int x1,y1;	/* shell's aft location */
	int x2,y2;	/* shell's nose location */
	int xl,yl;	 /* shell's left tip location */
	int xr,yr;	 /* shell's right tip location */
	int Right_Tip_Headings;
	int Left_Tip_Headings;
//	int svcolor;

//	svcolor=getcolor(); /* save present color */
//	setcolor(SHELL_COLOR);
	#ifdef GUI
	cairo_set_source_rgb(cr, 1.0, 0.0, 0.0);
	#else
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.2126);
	#endif
	
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 1.0, 0.0, 0.0);
	#endif
	x1=x;
	y1=y;
	x2=x1+size*Fsin(Headings);
	y2=y1-size*Fcos(Headings);
	Right_Tip_Headings=Headings+30;
	if(Right_Tip_Headings>359) Right_Tip_Headings=Right_Tip_Headings-360;
	Left_Tip_Headings=Headings+330;
	if(Left_Tip_Headings>359) Left_Tip_Headings=Left_Tip_Headings-360;
	xr=x1+0.4*size*Fsin(Right_Tip_Headings);
	yr=y1-0.4*size*Fcos(Right_Tip_Headings);
	xl=x1+0.4*size*Fsin(Left_Tip_Headings);
	yl=y1-0.4*size*Fcos(Left_Tip_Headings);
	cairo_move_to(cr,x1,y1);
	cairo_line_to(cr,xl,yl);
	cairo_line_to(cr,x2,y2);
	cairo_line_to(cr,xr,yr);
	cairo_line_to(cr,x1,y1);
//	PrevShell = cairo_copy_path(cr);
//	setcolor(svcolor); /* restore previous color */
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
	game_iteration();
	update_drawing(SF_canvas);
	#ifdef GUI_INTERFACE
	update_drawing(SF_rgb_context);
	#endif

	if(Jitter_Flag)
	{
		if(jitter_switch)
		{
				jitter_switch = 0;
		}
		else
		{
			Jitter_Step--;
			jitter_switch = 1;
		}
	}
	else if(Explosion_Flag)
	{
		Explosion_Step++;
	}

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

void jitter_step1(cairo_t *cr, int step)
{
	Ship_Headings=Ship_Headings+2*step;
	Ship_X_Pos=Ship_X_Old_Pos+step*Fcos(Ship_Headings);
	Ship_Y_Pos=Ship_Y_Old_Pos+step*Fsin(Ship_Headings);
}

void jitter_step2(cairo_t *cr, int step)
{
 	Ship_Headings=Ship_Headings-2*step;
  Ship_X_Pos=Ship_X_Old_Pos+step*Fsin(Ship_Headings);
  Ship_Y_Pos=Ship_Y_Old_Pos+step*Fcos(Ship_Headings);
}


void explosion_step1(cairo_t *cr, int X_Pos,int Y_Pos, int step)
{
  int i = 10 * (step + 1);
  int iarc;


	cairo_set_source_rgba(cr, 0, 0, 0, 0.2126);
	#ifdef GUI_INTERFACE
	cairo_set_source_rgb(SF_rgb_context, 1.0, 0.0, 0.0);
	#endif
//	cairo_set_source_rgba(cr, 0.0, 1, 0);
	for(iarc=i/5;iarc<360+i/5;iarc=iarc+20)
  {
		cairo_new_sub_path(cr);
		cairo_arc(cr, X_Pos,Y_Pos, i, deg2rad(iarc), deg2rad(iarc+2));
	}
	stroke_in_clip(cr);
//	cairo_reset_clip(cr);
//	cairo_stroke(cr);
}

void explosion_step2(cairo_t *cr, int X_Pos,int Y_Pos, int step)
{
	int j = step * 10;
  int iarc;

	if (j>0)
	{
		cairo_set_source_rgba(cr, 0, 0, 0, 0.9278);
		#ifdef GUI_INTERFACE
		cairo_set_source_rgb(SF_rgb_context, 1.0, 1.0, 0.0);
		#endif
		for(iarc=j/5;iarc<360+j/5;iarc=iarc+20)
		{
			cairo_new_sub_path(cr);
			cairo_arc(cr,X_Pos,Y_Pos,j,deg2rad(iarc),deg2rad(iarc+2));
		}
		stroke_in_clip(cr);
//		cairo_reset_clip(cr);
//		cairo_stroke(cr);
	}
}


void Reset_Screen()
{
        /*  reset variables */
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
  Loop_Counter = 0;
  Last_Missile_Hit = -11;
  Score=0.0;
  Ship_X_Speed=0.0;
  Ship_Y_Speed=0.0;
  Ship_Headings=0;
  srand(time(NULL));
	Ship_Damaged_By_Fortress = 0;
  Mine_Flag=DEAD;
  for(int i=0;i<MAX_NO_OF_MISSILES;i++) Missile_Flag[i]=DEAD;
  Missile_Type=VS_FRIEND;
  Missile_Vs_Mine_Only=OFF;
  Missiles_Counter=0;
  Shell_Flag=DEAD;
  Rotate_Input=0; /* joystick left/right */
  Accel_Input=0; /* joystick forward */
  End_Flag=OFF;
  Fort_Headings=270;
  Vulner_Counter=0;
  Timing_Flag=OFF; /* if screen reset between consecutive presses */
  Resource_Flag=OFF;
  Resource_Off_Counter=0;
  Bonus_Display_Flag=NOT_PRESENT;   /* in case bonus is pressed after */
  Bonus_Granted=OFF;
  Fort_Lock_Counter=0;
	// Score=0;
	// Points=0;
	// Velocity=0;
	// Control=0;
	// Speed=0;

}  /* end reset screen */


#endif
