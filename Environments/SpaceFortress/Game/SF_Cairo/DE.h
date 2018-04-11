#include <cairo.h>

#ifdef GUI
#define TEXT_HEIGHT 8/SCALE_F // The height of character "h" in pixels in Cairo (with monospace font)
#define TEXT_WIDTH 8/SCALE_F // The width of character "z" in pixels (with monospace font)
#else
#define TEXT_HEIGHT 6/SCALE_F
#define TEXT_WIDTH 8/SCALE_F
#endif
#define SF_YELLOW 1.0, 1.0, 0.33
#define SF_GREEN 0.0, 0.66, 0.0
#define SF_BLUE 0.33, 1.0, 1.0
#define SF_ORANGE 1.0, 0.33, 0.33
#define WINDOW_HEIGHT 480
#define WINDOW_WIDTH 640

float deg2rad(int deg);
void jitter_step1(cairo_t *cr, int step);
void jitter_step2(cairo_t *cr, int step);
void Reset_Screen(cairo_t *cr);

void Initialize_Graphics(cairo_t *cr);

void Close_Graphics(cairo_t *cr);
void Close_Graphics_SF();

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
void Draw_Fort(cairo_t *cr, int x, int y, int Headings, int size);
void Draw_Mine (cairo_t *cr, int x, int y, int size);
void Draw_Missile (cairo_t *cr, int x, int y, int Headings, int size, int missile_idx);
void Draw_Shell(cairo_t *cr, int x, int y, int Headings, int size);

float Find_Headings(int x1, int y1, int x2, int y2);

void set_initial_vals(cairo_t *cr);
void start_drawing();
void set_key(int key_value);
void stop_drawing();

int move_update();
void update_frame(cairo_t *cr);
unsigned char* update_frame_SF();
void stroke_in_clip(cairo_t *cr);

void Show_Score(cairo_t *cr, int val, int x, int y, int erease);

void Update_Points(cairo_t *cr, int earese);
void Update_Control(cairo_t *cr, int earese);
void Update_Velocity(cairo_t *cr, int earese);

void Update_Vulner(cairo_t *cr, int earese);
void Update_Interval(cairo_t *cr, int earese);
void Update_Speed(cairo_t *cr, int earese);
void Update_Shots(cairo_t *cr, int earese);

void explosion_step1(cairo_t *cr, int X_Pos,int Y_Pos,int step);
void explosion_step2(cairo_t *cr, int X_Pos,int Y_Pos,int step);


//static gboolean on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data);
//void animation_loop(GtkWidget *darea);