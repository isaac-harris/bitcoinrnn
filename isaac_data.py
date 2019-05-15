def drawShape(type, fill_red, fill_green, fill_blue, fill_opa, stroke_red, stroke_green, stroke_blue, stroke_opa, x_pos, y_pos, x_len, y_len):
    if type == "rectangle":
        fill(fill_red, fill_green, fill_blue, fill_opa)
        stroke(stroke_red, stroke_green, stroke_blue, stroke_opa)
        rect(x_pos, y_pos, x_len, y_len)
    elif type == "ellipse":
        fill(fill_red, fill_green, fill_blue, fill_opa)
        stroke(stroke_red, stroke_green, stroke_blue, stroke_opa)
        ellipse(x_pos, y_pos, x_len, y_len)
        
class circle(object):
    def __init__(self):
        self.x_coord = 320
        self.y_coord = 180
        self.up = 0
        self.down = 0
        self.red_val = 70
        self.green_val = 70
        self.blue_val = 70
        self.dia = 20
    
    
    def refresh(self):
        drawShape("rectangle",255,255,127,255,255,255,127,255,0,180,320,255)
        drawShape("rectangle",127,255,255,255,127,255,255,255,0,240,180,120)
        drawShape("ellipse",255,150,150,127,70,70,70,255,self.x_coord,self.y_coord,self.dia,self.dia)

def setup():
    size(640,360)
    global c
    c = circle()
    #fill(c.red_val,c.green_val,c.blue_val)
    #noFill()
    
    
    
def draw():
    frameRate(100)
    background(240,240,240)
    if mousePressed == True:
        c.x_coord = mouseX
        c.y_coord = mouseY
    c.refresh()

def keyPressed():
    if keyCode == UP and c.dia < 200:
        c.dia += 5
    if keyCode == DOWN and c.dia > 0:
        c.dia -= 5
    if key == "k":
        c.x_coord = mouseX
        c.y_coord = mouseY
    
    if key == "w":
        c.y_coord -= 5
    if key == "s":
        c.y_coord += 5
    if key == "a":
        c.x_coord -= 5
    if key == "d":
        c.x_coord += 5
