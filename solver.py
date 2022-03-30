import sys
import collections
import numpy as np
import heapq
import time
import numpy as np

# HiiImKinn: Using the Python Counter tool, you can count the key-value pairs in an object
from collections import Counter

global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum - colsNum)])

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""

    # Vị trí khởi đầu của các box khi bắt đầu map
    beginBox = PosOfBoxes(gameState)

    # Vị trí khởi đầu của Sokoban (nhân vật điều khiển) khi bắt đầu map
    beginPlayer = PosOfPlayer(gameState)

    # Trạng thái bắt đầu game (vị trí Sokoban, vị trí các box)
    startState = (beginPlayer, beginBox)

    # Hàng đợi chứa các node trạng thái chưa được xử lí
    frontier = collections.deque([[startState]])

    # Tập chứa các node trạng thái đã xảy ra, lưu dưới dạng tập set() để bỏ qua giai đoạn kiểm tra trạng thái đã xảy ra hay chưa
    exploredSet = set()

    # Hàng đợi chứa chuỗi hành động của Sokoban
    actions = [[0]] 

    # Danh sách lưu trữ đường đi kết quả của thuật toán
    temp = []

    # Kiểm tra hàng đợi còn các node trạng thái chưa được xử lí hay không
    while frontier:

        # Lấy ra trạng thái và hành động nằm cuối lần lượt nằm trong mỗi hàng đợi
        node = frontier.pop()
        node_action = actions.pop()

        # Kiểm tra nếu trạng thái hiện tại là trạng thái cuối cùng thì kết thúc và chuyển map
        if isEndState(node[-1][-1]):
            print(node[-1][-1])

            # Lưu kết quả và kết thúc vòng lặp while
            temp += node_action[1:]
            break

        # kiểm tra liệu trạng thái hiện tại đã được thực hiện chưa
        if node[-1] not in exploredSet:

            # Nếu trạng thái chưa thực hiện, thêm trạng thái vào tập trạng thái đã xảy ra
            exploredSet.add(node[-1])

            # Vòng lặp duyệt các hành động có thể thực thi từ trạng thái hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):

                # Vị trí mới của Sokoban và các box trong hành động kế tiếp đó
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Vị trí box rơi vào trường hợp không hợp lệ (xuyên tường) thì bỏ qua
                if isFailed(newPosBox): continue

                # Thêm trạng thái và hành động vào mỗi hàng đợi tương ứng
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

    # Trả về kết quả đường đi lưu vào danh sách [temp]
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""

    # Vị trí khởi đầu của các box khi bắt đầu map
    beginBox = PosOfBoxes(gameState)

    # Vị trí khởi đầu của Sokoban (nhân vật điều khiển) khi bắt đầu map
    beginPlayer = PosOfPlayer(gameState)

    # Trạng thái bắt đầu game (vị trí Sokoban, vị trí các box)
    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    
    # Hàng đợi chứa các node chưa được xử lí
    frontier = collections.deque([[startState]]) # store states

    # Hàng đợi chứa chuỗi hành động của Sokoban
    actions = collections.deque([[0]]) # store actions

    # Tập chứa các node trạng thái đã xảy ra, lưu dưới dạng tập set() để bỏ qua giai đoạn kiểm tra trạng thái đã xảy ra hay chưa
    exploredSet = set()

    # Danh sách lưu trữ đường đi kết quả của thuật toán
    temp = []

    ### Implement breadthFirstSearch here
    
    # Kiểm tra hàng đợi còn các node trạng thái chưa được xử lí hay không
    while frontier:

        # Lấy ra trạng thái và hành động nằm đầu tiên lần lượt nằm trong mỗi hàng đợi
        node = frontier.popleft()
        node_action = actions.popleft()

        # Kiểm tra nếu trạng thái hiện tại là trạng thái cuối cùng thì kết thúc và chuyển map
        if isEndState(node[-1][-1]):

            # Lưu kết quả và kết thúc vòng lặp while
            temp = node_action[1:]
            return temp

        # kiểm tra liệu trạng thái hiện tại đã được thực hiện chưa
        if node[-1] not in exploredSet: 
            
            # Nếu trạng thái chưa thực hiện, thêm trạng thái vào tập trạng thái đã xảy ra
            exploredSet.add(node[-1])
            
            # Vòng lặp duyệt các hành động có thể thực thi từ trạng thái hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):

                # Vị trí mới của Sokoban và các box trong hành động kế tiếp đó
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Vị trí box rơi vào trường hợp không hợp lệ (xuyên tường) thì bỏ qua
                if isFailed(newPosBox): continue

                # Thêm trạng thái và hành động vào mỗi hàng đợi tương ứng
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

    # Trả về kết quả đường đi lưu vào danh sách [temp]
    return temp

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    
    # Vị trí khởi đầu của các box khi bắt đầu map
    beginBox = PosOfBoxes(gameState)

    # Vị trí khởi đầu của Sokoban (nhân vật điều khiển) khi bắt đầu map
    beginPlayer = PosOfPlayer(gameState)

    # Trạng thái bắt đầu game (vị trí Sokoban, vị trí các box)
    startState = (beginPlayer, beginBox)

    # Hàng đợi chứa các node chưa được xử lí, sử dụng cấu trúc dữ liệu Priority Queue để đưa các node có chi phí thấp nhất lên đầu hàng đợi
    frontier = PriorityQueue()
    frontier.push([startState], 0)

    # Tập chứa các node trạng thái đã xảy ra, lưu dưới dạng tập set() để bỏ qua giai đoạn kiểm tra trạng thái đã xảy ra hay chưa
    exploredSet = set()
    
    # Hàng đợi chứa chuỗi hành động của Sokoban
    actions = PriorityQueue()
    actions.push([0], 0)

    # Danh sách lưu trữ đường đi kết quả của thuật toán
    temp = []

    ### Implement uniform cost search here

    # Kiểm tra hàng đợi còn các node trạng thái chưa được xử lí hay không
    while not frontier.isEmpty():
        
        # Lấy ra trạng thái và hành động nằm trên đỉnh lần lượt nằm trong mỗi hàng đợi (min heap)
        node = frontier.pop()
        node_action = actions.pop()

        # Kiểm tra nếu trạng thái hiện tại là trạng thái cuối cùng thì kết thúc và chuyển map
        if isEndState(node[-1][-1]):

            # Lưu kết quả và kết thúc vòng lặp while
            temp = node_action[1:]
            return temp
            
        # kiểm tra liệu trạng thái hiện tại đã được thực hiện chưa
        if node[-1] not in exploredSet:

            # Nếu trạng thái chưa thực hiện, thêm trạng thái vào tập trạng thái đã xảy ra
            exploredSet.add(node[-1])

            # Vòng lặp duyệt các hành động có thể thực thi từ trạng thái hiện tại
            for action in legalActions(node[-1][0], node[-1][1]):

                # Vị trí mới của Sokoban và các box trong hành động kế tiếp đó
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Vị trí box rơi vào trường hợp không hợp lệ (xuyên tường) thì bỏ qua
                if isFailed(newPosBox): continue

                # Thêm trạng thái và hành động vào mỗi hàng đợi tương ứng, chi phí tính toán là tổng chi phí node đang xét và chi phí đã tính trước đó
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(node_action[1:] + [action[-1]]))
                actions.push(node_action + [action[-1]], cost(node_action[1:] + [action[-1]]))
    
    # Trả về kết quả đường đi lưu vào danh sách [temp]
    return temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels, "r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)

    # Trả về kết quả đường đi của giải thuật và trả về thời gian tính toán giải thuật đó
    return [result, time_end - time_start]
