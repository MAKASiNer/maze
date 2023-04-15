import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1' # сокрытие pygame prompt

import pygame
import argparse
from maze import MazeDFS, MazeMST


def build(w, h, method, output=None):
    if w < 3 or h < 3:
        print('The side of the maze cannot be less than 3')
        exit(1)

    m = method.upper()

    if m == 'DFS':
        maze = MazeDFS(w, h)
        ext = '.dfs'
    elif m == 'MST':
        maze = MazeMST(w, h)
        ext = '.mst'
    else:
        print(f"Unsupported generation method '{m}'")
        exit(1)

    maze.build()

    if output is not None:
        maze.save(output + ext)

    return maze


def _open_maze(input):
    path, _, ext = input.rpartition('.')

    if not path:
        print("Сan't identify method of maze generation")
        exit(1)

    if ext == 'dfs':
        MazeClass = MazeDFS
    elif ext == 'mst':
        MazeClass = MazeMST
    else:
        print(f"Unsupported file extension '{ext}'")
        exit(1)

    try:
        return MazeClass.load(input)
    except FileNotFoundError:
        print(f"No such file or directory '{input}'")
        exit(1)


def show(input, s):
    maze = _open_maze(input)
    solution = maze.solve() if s else None
    return maze.to_str(solution, '██', '  ', '. ')


def play(input):
    maze = _open_maze(input)

    map = maze.to_map()
    w, h = len(map[0]), len(map)
    cell = 20
    speed = 0.15

    # отрисовка фона на поверхность
    def draw_maze(surface):
        # лабиринт
        for y in range(h):
            for x in range(w):
                clr = '#000000' if map[y][x] else '#ffffff'
                pygame.draw.rect(surface, clr, [x * cell, y * cell, cell, cell])

        # флагшток
        pnts = [(cell * (w - 1.7), cell * (h - 1.9)),
                (cell * (w - 1.8), cell * (h - 1.9)),
                (cell * (w - 1.8), cell * (h - 1.1)),
                (cell * (w - 1.7), cell * (h - 1.1))]

        pygame.draw.polygon(surface, '#000000', pnts)

        # флаг
        pnts = [(cell * (w - 1.8), cell * (h - 1.9)),
                (cell * (w - 1.2), cell * (h - 1.9)),
                (cell * (w - 1.2), cell * (h - 1.6)),
                (cell * (w - 1.8), cell * (h - 1.6))]

        pygame.draw.polygon(surface, '#f36060', pnts)
        pygame.draw.polygon(surface, '#000000', pnts, width=1)

    # перемещение по координатам
    def goto(xy, delta):
        x, y = xy
        dx, dy = delta

        if 0 < (x + dx) // cell < w and 0 < (y + dy) // cell < h:
            if not map[round(y + dy) // cell][round(x + dx) // cell]:
                return x + dx, y + dy
        
        return x, y

    pygame.init()
    screen = pygame.display.set_mode((800, 600), vsync=1)
    pygame.display.set_caption('Maze')

    bg = pygame.Surface((w * cell, h * cell))
    draw_maze(bg)

    x, y = int(cell * 1.5), int(cell * 1.5)
    camera = pygame.Rect(0, 0, screen.get_width(), screen.get_height())

    game_running = True
    while game_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_running = False
                
        # проверка нажатых кнопок
        dx, dy = 0, 0
        pressed_keys = pygame.key.get_pressed()

        shift = speed
        if pressed_keys[pygame.K_LSHIFT] or pressed_keys[pygame.K_RSHIFT]:
            shift = speed * 3
            
        if pressed_keys[pygame.K_LEFT]:
            dx += -shift
        elif pressed_keys[pygame.K_RIGHT]:
            dx += shift
        elif pressed_keys[pygame.K_UP]:
            dy += -shift
        elif pressed_keys[pygame.K_DOWN]:
            dy += shift

        # перемещение персонажа (если это возможно)
        x, y = goto((x, y), (dx, dy))

        # смещение камеры так, чтобы персонаж оставался в центре
        camera.x = x - camera.width / 2
        camera.y = y - camera.height / 2

        # отрисовка фона и персонажа
        screen.blit(bg, (0, 0), camera)
        pygame.draw.circle(screen, '#60f3f3', screen.get_rect().center, cell // 4)
        pygame.draw.circle(screen, '#000000', screen.get_rect().center, cell // 4, width=1)
        pygame.display.update()

        if x // cell == w - 2 and  y // cell == h - 2:
            game_running = False
            print('You win!!!')

    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(title='commands', dest='cmd')

    cmd_build = subparser.add_parser('build', help='creates a new maze')
    cmd_build.add_argument('w', type=int, help='maze width')
    cmd_build.add_argument('h', type=int, help='maze height')
    cmd_build.add_argument('-m', '--method', metavar='', required=True, help='generation method (available DFS or MST)')
    cmd_build.add_argument('-o', '--output', metavar='', required=False, help='path to save')

    cmd_show = subparser.add_parser('show', help='prints maze to console')
    cmd_show.add_argument('-i', '--input', metavar='', required=True, help='path to file')
    cmd_show.add_argument('-s', action='store_true', default=False, help='solve the maze')

    cmd_play = subparser.add_parser('play', help='go over maze')
    cmd_play.add_argument('-i', '--input', metavar='', required=True, help='path to file')

    args = parser.parse_args()

    cmd = args.cmd

    if cmd == 'build':
        build(args.w, args.h, args.method, args.output)
        exit(0)

    elif cmd == 'show':
        print(show(args.input, args.s))
        exit(0)

    elif cmd == 'play':
        play(args.input)
        exit(0)
