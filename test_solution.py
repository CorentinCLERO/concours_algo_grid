import json
import math

SUBMISSION_FILE = 'solutions/1_example_submission.txt'
DATASET_FILE = 'datasets/1_example.json'

def get_solution_score(solution_txt, dataset_txt):
    """Evaluate the solution and return a tuple (score, isValid, error_message).

    Arguments:
    solution -- the solution to be evaluated
    dataset -- the dataset for which the solution is made
    """
    try:
        dataset = json.loads(dataset_txt)
    except:
        return 0, False, 'Error while processing the dataset. Please ensure you have the correct file.'
    
    target_grid = dataset['grid']
    max_actions = dataset['maxActions']
    max_jokers = dataset['maxJokers']
    max_joker_size = dataset['maxJokerSize']
    grid_height = len(target_grid)
    grid_width = len(target_grid[0])

    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    jokers_remaining = max_jokers

    try:
        solution_actions = solution_txt.splitlines()
    except:
        return 0, False, 'Submission is not valid.'
    
    if len(solution_actions) > max_actions:
        return 0, False, f'Too many actions: {len(solution_actions)} (maximum allowed: {max_actions})'
    
    for action in solution_actions:
        if action.startswith('RECT '):
            try:
                action_split = action.split()
                x1, y1, x2, y2, color = map(int, action_split[1:])
            except:
                return 0, False, f'Invalid RECT action: "{action}" (expected format: RECT <x1> <y1> <x2> <y2> <color>)'
            if not 0 <= x1 < grid_width or not 0 <= y1 < grid_height:
                return 0, False, f'Invalid RECT coordinates for point (x1, y1): "{action}" (coordinates out of bounds)'
            if not 0 <= x2 < grid_width or not 0 <= y2 < grid_height:
                return 0, False, f'Invalid RECT coordinates for point (x2, y2): "{action}" (coordinates out of bounds)'
            if x1 > x2 or y1 > y2:
                return 0, False, f'Invalid RECT coordinates: "{action}" (must have x1 <= x2 and y1 <= y2)'
            if not 0 <= color <= 7:
                return 0, False, f'Invalid RECT color: "{action}" (color must be between 0 and 7)'
            
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    grid[y][x] = color
        elif action.startswith('JOKER '):
            if jokers_remaining <= 0:
                return 0, False, f'Too many JOKER actions used (exceeds maximum allowed of {max_jokers})'
            jokers_remaining -= 1

            try:
                action_split = action.split()
                x1, y1, x2, y2 = map(int, action_split[1:])
            except:
                return 0, False, f'Invalid JOKER action: "{action}" (expected format: JOKER <x1> <y1> <x2> <y2>)'
            if not 0 <= x1 < grid_width or not 0 <= y1 < grid_height:
                return 0, False, f'Invalid JOKER coordinates for point (x1, y1): "{action}" (coordinates out of bounds)'
            if not 0 <= x2 < grid_width or not 0 <= y2 < grid_height:
                return 0, False, f'Invalid JOKER coordinates for point (x2, y2): "{action}" (coordinates out of bounds)'
            if x1 > x2 or y1 > y2:
                return 0, False, f'Invalid JOKER coordinates: "{action}" (must have x1 <= x2 and y1 <= y2)'

            joker_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if joker_area > max_joker_size:
                return 0, False, f'JOKER area too large: "{action}". Area {joker_area} ({x2 - x1 + 1} x {y2 - y1 + 1}) exceeds maximum allowed {max_joker_size})'

            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    grid[y][x] = target_grid[y][x]
        else:
            return 0, False, f'Invalid action: "{action}" (must start with RECT or JOKER)'
        
    nb_correct = sum(1 for y in range(grid_height) for x in range(grid_width) if grid[y][x] == target_grid[y][x])
    if nb_correct == grid_height * grid_width:
        score = round(max_actions / len(solution_actions) * 1_000_000)
        return score, True, f'Perfect solution! Used {len(solution_actions)} actions out of {max_actions} max. Score: {score}'
    
    score = round(nb_correct / (grid_height * grid_width) * 1_000_000)
    return score, True, f'Partial solution. Correct pixels: {nb_correct} out of {grid_height * grid_width}. Score: {score}'


if __name__ == '__main__':
    with open(SUBMISSION_FILE) as fi:
        solution = fi.read()
    with open(DATASET_FILE) as fi:
        dataset = fi.read()
    score, is_valid, message = get_solution_score(solution, dataset)

    if is_valid:
        print('✅ Solution is valid!')
        print(f'Score: {score:_}')
    else:
        print('❌ Solution is invalid')

    print(f'Message: {message}')
