#include <cairo.h>

#ifdef GUI
#define TEXT_HEIGHT 4 // The height of character "h" in pixels in Cairo (with monospace font)
#define TEXT_WIDTH 8 // The width of character "z" in pixels (with monospace font)
#else
#define TEXT_HEIGHT 10
#define TEXT_WIDTH 8
#endif
#define SF_YELLOW 0.0, 0.0, 0.0, 1
#define SF_GREEN 0.0, 0.0, 0.0, 0.4720
#define SF_BLUE 0.0, 0.0, 0.0, 0.66
#define SF_ORANGE 0.0, 0.0, 0.0, 0.4724
//#define WINDOW_WIDTH 240
//#define WINDOW_HEIGHT 240 + (TEXT_HEIGHT*3)
#define WINDOW_WIDTH 448 // The square width of the original game
//#define WINDOW_HEIGHT 448 + TEXT_HEIGHT*2
#define WINDOW_HEIGHT 448

float deg2rad(int deg);
void Open_Graphics(void);
void jitter_step1(cairo_t *cr, int step);
void jitter_step2(cairo_t *cr, int step);
void Reset_Screen();

float* get_symbols();
void Initialize_Graphics(cairo_t *cr);

unsigned char* update_screen();

void Close_Graphics(cairo_t *cr);
void Close_Graphics_SF();
void set_initial_vals();

float Fcos(int Headings_Degs);
float Fsin(int Headings_Degs);

void snapCoords(cairo_t *canvas, int x, int y);
void cairo_line(cairo_t *cr, int x1, int y1, int x2, int y2);
void cairo_text_at(cairo_t *cr, int x, int y, const char *string);

void clip_path_rect(cairo_t *cr);
void clear_prev_path(cairo_t *cr, cairo_path_t *prevPath);
void clean(cairo_t *cr);
void update_drawing(cairo_t *cr);

void Draw_Frame(cairo_t *cr);
void Draw_Hexagone(cairo_t *cr,int X_Center,int Y_Center,int Hex_Outer_Radius);
void Draw_Ship(cairo_t *cr, int x, int y, int Headings, int size);
void Draw_Fort(cairo_t *cr, int x, int y, int Headings, int size );
void Draw_Mine (cairo_t *cr, int x, int y, int size);
void Draw_Missile (cairo_t *cr, int x, int y, int Headings, int size, int missile_idx);
void Draw_Shell(cairo_t *cr, int x, int y, int Headings, int size);
void Draw_Ship_Nose(cairo_t *cr, int x, int y, int Headings, int size);


float Find_Headings(int x1, int y1, int x2, int y2);

//void set_initial_vals(cairo_t *cr);
void start_drawing();
void set_key(int key_value);
void stop_drawing();

int move_update();

unsigned char* update_frame();
void stroke_in_clip(cairo_t *cr);

void Show_Score(cairo_t *cr, int val, int x, int y);

void Update_Points(cairo_t *cr);
void Update_Control(cairo_t *cr);
void Update_Velocity(cairo_t *cr);

void Update_Vulner(cairo_t *cr);
void Update_Interval(cairo_t *cr);
void Update_Speed(cairo_t *cr);
void Update_Shots(cairo_t *cr);

void explosion_step1(cairo_t *cr, int X_Pos,int Y_Pos,int step);
void explosion_step2(cairo_t *cr, int X_Pos,int Y_Pos,int step);


//static gboolean on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data);
//void animation_loop(GtkWidget *darea);
