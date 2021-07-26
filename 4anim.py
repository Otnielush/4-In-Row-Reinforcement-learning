import numpy as np
from kivy.app import App

from kivy.config import Config
Config.set('graphics', 'resizable', 1)
Config.set('graphics', 'width', 400)
Config.set('graphics', 'height', 500)

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from random import (random, randint)
from kivy.core.window import Window
from kivy.graphics import (Color, Ellipse, Rectangle, Line)

from kivy.animation import Animation


def posMoves():
    posList = []
    for j in range(7):
        i = 5
        if not board[0][j] == " ":
            posList.append(-1)
            continue
        while not board[i][j] == " " and i > 0:
            i -= 1
        posList.append(i)
    return posList

# Function calculates combinations in a line
# both for the move and for checking
def maxInRow(player, row, col):
    inRow4 = 0
    inRow3 = 0
    inRow2 = 0
    inRowFree = 0
    numInRow = 0
    inRow = 0
    inRowPosib = 0
    inRowMax = 0
    sim = "O" if player == 0 else "#"
    rowTmp = board[row].copy()
    rowTmp[col] = sim
    start = col - 3
    if start < 0: start = 0
    end = col + 4
    if end > 7: end = 7

    for i in range(start, end):
        if rowTmp[i] == " ":
            inRowPosib += 1
            inRowFree += 1
            inRow = 0
            continue
        if rowTmp[i] == sim:
            inRow += 1
            inRowPosib += 1
            numInRow += 1
            if inRow > inRowMax: inRowMax = inRow
        else:
            inRowFree = 0
            inRow = 0
            inRowPosib = 0
            numInRow = 0

    else:
        if inRowMax >= 4: inRow4 = 1
        if inRowPosib >= 4:
            if inRowMax == 3 or numInRow >= 3 and inRowMax == 2 : inRow3 = 1
            if inRowMax == 2 or numInRow > 1: inRow2 = 1
    return inRow4, inRow3, inRow2, inRowFree

# Function calculated combination in column
# both for the move and for checking
def maxInCol(player, row, col):
    inCol4 = 0
    inCol3 = 0
    inCol2 = 0
    numInCol = 0
    inCol = 0
    inColPosib = 0
    inColMax = 0
    colTmp = []
    sim = "O" if player == 0 else "#"
    for rows in board:
        colTmp.append(rows[col])
    colTmp[row] = sim

    start = row + 4
    if start > 5: start = 5
    end = -1

    for i in range(start, end, -1):
        if colTmp[i] == " ":
            inColPosib += 1
            inCol = 0
            continue
        if colTmp[i] == sim:
            inCol += 1
            inColPosib += 1
            numInCol += 1
            if inCol > inColMax: inColMax = inCol
        else:
            inCol = 0
            inColPosib = 0

    else:
        if inColMax >= 4: inCol4 = 1
        if inColPosib >= 4:
            if inColMax == 3: inCol3 = 1
            if inColMax == 2: inCol2 = 1
    return inCol4, inCol3, inCol2, inColPosib - numInCol

# Functions calculate diagonal combinations
# \
def maxInDiagLeft(player, row, col):
    inDiag4 = 0
    inDiag3 = 0
    inDiag2 = 0
    numInDiag = 0
    inDiag = 0
    inDiagPosib = 0
    inDiagMax = 0
    diagTmp = []
    sim = "O" if player == 0 else "#"
    for i in range(6):
        cols = col - (row - i)
        if cols < 0: continue
        if cols > 6: break
        diagTmp.append(board[i][cols])
        if row == i: diagTmp[len(diagTmp) - 1] = sim

    diagTmp.reverse()
    for cell in diagTmp:
        if cell == " ":
            inDiagPosib += 1
            inDiag = 0
            continue
        if cell == sim:
            inDiag += 1
            inDiagPosib += 1
            numInDiag += 1
            if inDiag > inDiagMax: inDiagMax = inDiag
        else:
            inDiag = 0
            inDiagPosib = 0

    else:
        if inDiagMax >= 4: inDiag4 = 1
        if inDiagPosib >= 4:
            if inDiagMax == 3 or inDiag >= 1 and inDiagMax >= 2: inDiag3 = 1
            if inDiagMax == 2 or inDiag == 1 and numInDiag > 1: inDiag2 = 1
        if inDiagPosib >= 5:
            if inDiag == 1 and numInDiag >= 2: inDiag2 += 1
    return inDiag4, inDiag3, inDiag2, inDiagPosib - numInDiag

# /
def maxInDiagRight(player, row, col):
    inDiag4 = 0
    inDiag3 = 0
    inDiag2 = 0
    numInDiag = 0
    inDiag = 0
    inDiagPosib = 0
    inDiagMax = 0
    diagTmp = []
    sim = "O" if player == 0 else "#"
    for i in range(6):
        cols = col + (row - i)
        if cols < 0: break
        if cols > 6: continue
        diagTmp.append(board[i][cols])
        if row == i: diagTmp[len(diagTmp) - 1] = sim

    diagTmp.reverse()
    for cell in diagTmp:
        if cell == " ":
            inDiagPosib += 1
            inDiag = 0
            continue
        if cell == sim:
            inDiag += 1
            inDiagPosib += 1
            numInDiag += 1
            if inDiag > inDiagMax: inDiagMax = inDiag
        else:
            inDiag = 0
            inDiagPosib = 0

    else:
        if inDiagMax >= 4: inDiag4 = 1
        if inDiagPosib >= 4:
            if inDiagMax == 3 or inDiag >= 1 and inDiagMax >= 2: inDiag3 = 1
            if inDiagMax == 2 or inDiag == 1 and numInDiag > 1: inDiag2 = 1
        if inDiagPosib >= 5:
            if inDiag == 1 and numInDiag >= 2: inDiag2 += 1
    return inDiag4, inDiag3, inDiag2, inDiagPosib - numInDiag

# Checking if there are 4 characters in a row to award victory
# Checking after move
def win(row, col):
    # print("Win check")
    global wins
    winner = "Homo sapiens" if currPlayer == 0 else "Artificial Intelligence"
    for i in range(len(Review)):
        result = Review[i](currPlayer, row, col)
        if result[0] >= 1:
            # print("Winner by {} - {}".format(winType[i], winner))
            wins[currPlayer] += 1
            # network.rewards.append(1 if currPlayer == 1 else 0)
            txt = "{}:{}".format(wins[0], wins[1])
            GraphGame.set_score(txt)
            # print("Score of epic competition:\n Human: %d  AI: %d" % (wins[0], wins[1]))
            return 1

# The function returns the sum of all combinations for a specific move.
def makeValuation(player, row, col):
    global wins
    four, three, two, possib = 0, 0, 0, 0
    for i in range(len(Review)):
        n4, n3, n2, p = Review[i](player, row, col)
        four += n4
        three += n3
        two += n2
        possib += p
    # print("{} four {}, thr {}, two {}, po {}-{}".format(col+1, four, three, two, possib,player))
    return four, three, two, possib

def newGame():
    global board
    board = [[" "] * 7 for i in range(6)]

def calcWeights():
    global movePoints
    moveList = posMoves()

    for i in range(len(moveList)):
        floor = 1
        if moveList[i] == -1:
            movePoints[0][i] = -1
            movePoints[1][i] = -1
            continue
        n4H, n3H, n2H, pH = makeValuation(0, moveList[i], i)  # Human/opponent moves profit
        n4, n3, n2, p = makeValuation(1, moveList[i], i)  # AI moves profit
        floor -= (sum(moveList) / len(moveList) - moveList[i]) * 0.15
        if floor > 1: floor = 1
        movePoints[0][i] = n4H * weights[4] + (n3H * weights[5] + n2H * weights[6] + pH * weights[7]) * floor
        movePoints[1][i] = n4 * weights[0] + (n3 * weights[1] + n2 * weights[2] + p * weights[3]) * floor
    return moveList

# AI move. We get a list of possible moves,
# calculate the usefulness of each of them
# and return the number of columns where we will go
def AImove(myNum):
    global agr, agrTurns
    enemy = (myNum + 1) % 2

    stepList = [0]*7
    for i in range(7):
        stepList[i] = movePoints[myNum][i] + movePoints[enemy][i]*agr

    if agr > 0.85: agrTurns -= 1
    if agrTurns <= 0:
        # print("Agr stopped")
        agr = 0.85
        agrTurns = strategy[1]
    # print("{}".format(stepList))
    return stepList.index(max(stepList))

# CNN
# свой знак на 0 из 2
# myNum: 0 - player, 1 - AI
from FourInRow_torch import ActorCritic
network = ActorCritic('n', training=True, obs_shape=(6, 7, 2), act_shape=7, buffer_size=100, lr=0.001)
def AImove(myNum):
    inp = np.zeros((6, 7, 2))
    brd = np.array(board)
    if myNum == 0:
        my_sym = "O"
        en_sym = '#'
    else:
        my_sym = "#"
        en_sym = 'O'

    inp[:,:, 0] = (brd == my_sym) * 1
    inp[:,:,1] = (brd == en_sym) * 1
    mask = ((inp[0,:,0] + inp[0,:,1]) < 1) * 1
    action = network.select_action(inp, mask)
    # print(board)
    # print(inp)
    # print(mask)

    return action


# 4 functions for analise in 1 massive
Review = [maxInRow, maxInCol, maxInDiagLeft, maxInDiagRight]
winType = ["row", "column", "left diagonal", "right diagonal"]

board = [[" "] * 7 for i in range(6)] # [rows][columns]
currPlayer = 0
wins = [0, 0]


# Strategies with weights
# 1st - Multiplier against Human after agression, 2nd - how many turns
# Tit-for-tat – מכה מול מכה
Tit4Tat = [2, 1]
# Grim-Trigger – פוגע חזרה ולא סולח
GrimTrigger = [2, 200]
# Forgiving Trigger – פוגע חזרה וסולח
Forgiving = [2, 3]

# What strategy does AI use for aggression
strategy = Forgiving
agr = 0.85
agrTurns = strategy[1]
# Variables
choiceCl = 0
weights = [400, 100, 35, 3, 400, 100, 35, 3]
movePoints = [[0]*7 for i in range(2)]


class BGLines(Widget):
    def __init__(self, **kwargs):
        super(BGLines, self).__init__(**kwargs)
        with self.canvas:
            botSkelet = Window.size[1] * 0.2
            rightX = Window.size[0] * .9
            leftX = Window.size[0] * .1
            topSkelet = botSkelet + Window.size[0] * .75
            widX = (Window.size[0] - leftX * 2) / 7
            heigY = (topSkelet - botSkelet) / 6
            # print(topSkelet, botSkelet, rightX, leftX, widX)
            Color(46 / 255, 30 / 255, 38 / 255)
            Line(points=(leftX, topSkelet, leftX, botSkelet, rightX, botSkelet, rightX, topSkelet), width=2)
            # Vertical lines
            for x in range(6):
                Line(points=(leftX + widX * (x + 1), topSkelet, leftX + widX * (x + 1), botSkelet), width=1.5)
            # Horizontal lines
            for y in range(6):
                Line(points=(leftX, topSkelet - heigY * y, rightX, topSkelet - heigY * y), width=1)



class Painter(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.botSkelet = Window.size[1] * 0.2
        self.leftX = Window.size[0] * .1
        topSkelet = self.botSkelet + Window.size[0] * .75
        self.widX = (Window.size[0] - self.leftX * 2) / 7
        self.heigY = (topSkelet - self.botSkelet) / 6

        self.winning = False
        self.animBlock = False
        self.AI = False

    def block(self):
        self.animBlock = True

    def antiBlock(self):
        if self.winning:
            return

        if self.AI:
            moveList = calcWeights()
            col = AImove(1)
            row = moveList[col]
            self.make_move(row,col)


        self.animBlock = False

    ball = None


    def make_move(self,row,col):
        global currPlayer, board
        with self.canvas:
            if currPlayer == 0:
                Color(.1, .1, .8)
                sim = "O"
                # print('Player move, "O"')
            else:
                Color(.8, .8, .1)
                sim = "#"
                # print('Ai move, "#"')

            rad = Window.size[0]*0.1

            xp = self.leftX+self.widX/2+(self.widX*col)-rad/2
            yp = self.botSkelet+3+self.heigY*(5-row)
            board[row][col] = sim

            # Check for winning instanse
            if 1 == win(row, col):
                self.winning = True
                network.rewards.append(1 if currPlayer == 1 else -1)
                # network training
                network.finish_episode()
                GraphGame.show_but()
            else:
                network.rewards.append(0)

            currPlayer = (currPlayer+1) % 2
            self.AI = True if currPlayer == 1 else False

            self.ball = Ellipse(pos=(xp, Window.size[1] + 20), size=(rad, rad))
        self.animus(self.ball, xp, yp)


    def on_touch_up(self, touch):
        if self.animBlock or self.AI or self.winning: return
        self.animBlock = True
        global choiceCl
        posList = posMoves()
        col = 0
        tchX = touch.x
        if tchX > self.leftX:
            tchX -= self.leftX
            col = int(tchX) // int(self.widX)
            if col > 6: col = 6

        choiceCl = col
        if not board[0][col] == " ":
            # print("Column full. Please choose another")
            self.animBlock = False
            return
        self.AI = True
        self.make_move(posList[col],col)


    def animus(self, obj, xp, yp):
        rad = Window.size[0] * 0.1
        anim = Animation(pos=(xp, yp), duration=(Window.size[1] / 800), transition="in_cubic")
        anim += Animation(size=(rad, rad-6), duration=0.04)
        anim += Animation(size=(rad, rad), duration=0.02)
        anim += Animation(pos=(xp, yp+7), size=(rad, rad-4), duration=0.13, transition="out_cubic")
        anim += Animation(size=(rad, rad), duration=0.04)
        anim += Animation(pos=(xp, yp), duration=0.20, transition="in_cubic")
        # anim.repeat = True
        anim.on_start = lambda x: self.block()
        anim.on_complete = lambda x: self.antiBlock()
        anim.start(obj)




class Graphic(App):

    def set_score(self, txt):
        self.score.text = txt

    def build(self):
        lines = BoxLayout(orientation='vertical', padding=[5])

        self.score = (Label(text="0:0", font_size=25, size_hint=(.2, 1), halign="center", valign="bottom"))

        with lines.canvas.before:
            Color(156/255,144/255,187/255)
            self.rect = Rectangle(size=Window.size, pos=lines.pos)

        topLine = BoxLayout(orientation='horizontal', spacing=5, size_hint=(1, 0.05))

        topLine.add_widget(Label(text='[color=0000FF]Player[/color]', font_size=20,  halign="right", size_hint=(0.2, 1), markup = True))
        topLine.add_widget(self.score)
        topLine.add_widget(Label(text='[color=FFFF00]Machine[/color]', font_size=20, halign="left", size_hint=(0.25, 1), markup = True))

        al = AnchorLayout( size_hint=(0.1, 1), anchor_x='right', anchor_y='top')
        al.add_widget(Button(text='X', on_release=exit, size_hint=(None,None), size=(25,25)))
        topLine.add_widget(al)


        self.balls = Painter(size_hint=(.84,0.85))
        midLine = BGLines(size_hint=(.84,0.85))
        midLine.add_widget(self.balls)


        # botLine = Widget(size_hint=(1,0.1))
        self.botLine = Button(text="new game", on_release=self.new_game, size_hint=(1,0.1), disabled=True)

        lines.add_widget(topLine)
        lines.add_widget(midLine)
        lines.add_widget(self.botLine)

        return lines


    def hide_but(self):
        self.botLine.disabled = True

    def show_but(self):
        self.botLine.disabled = False

    def clear_canvas(self, instance):
        self.balls.canvas.clear()

    def new_game(self, instance):
        self.clear_canvas(instance)
        newGame()
        self.balls.winning = False
        self.balls.AI = False
        self.balls.antiBlock()
        self.hide_but()



def sizes(self):
    topSkelet = Window.size[1] - 40
    botSkelet = Window.size[1] * 0.1
    rightX = Window.size[0] * .84
    print(topSkelet, botSkelet, rightX)

GraphGame = Graphic()

if __name__ == "__main__":
    GraphGame.run()