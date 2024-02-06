import numpy as np

# get_objects(grid,diag=False,by_row=False,by_col=False,by_color=False,multicolor=False,more_info = True):
#   Takes in grid, returns list of object dictionary: top-left coordinate of object (’tl’), 2D grid (’grid’) by_row views splits objects
#   by grid rows, by_col splits objects by grid columns, by_color groups each color as one object, multicolor means object
#   can be more than one color. Empty cells within objects are represented as ’$’. If more_info is True, also returns size of
#   grid (’size’), cells in object (’cell_count’), shape of object (’shape’)
def get_objects(grid, diag=False, by_row=False, by_col=False, by_color=False, multicolor=False, more_info=True):
    objects = []
    visited = [[False]*len(grid[0]) for _ in range(len(grid))]

    def dfs(i, j, color, obj):
        if i<0 or i>=len(grid) or j<0 or j>=len(grid[0]) or visited[i][j] or grid[i][j]!=color:
            return
        visited[i][j] = True
        obj.append((i, j))
        dfs(i+1, j, color, obj)
        dfs(i-1, j, color, obj)
        dfs(i, j+1, color, obj)
        dfs(i, j-1, color, obj)
        if diag:
            dfs(i+1, j+1, color, obj)
            dfs(i-1, j-1, color, obj)
            dfs(i+1, j-1, color, obj)
            dfs(i-1, j+1, color, obj)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not visited[i][j] and grid[i][j] != '.':
                obj = []
                dfs(i, j, grid[i][j], obj)
                row_bounds = (min(x for x, _ in obj), max(x for x, _ in obj))
                col_bounds = (min(y for _, y in obj), max(y for _, y in obj))
                obj_grid = [row[col_bounds[0]:col_bounds[1]+1] for row in grid[row_bounds[0]:row_bounds[1]+1]]
                obj_dict = {'tl': (i, j), 'grid': obj_grid}
                if more_info:
                    obj_dict['size'] = (len(grid), len(grid[0]))
                    obj_dict['cell_count'] = sum(cell != '.' for row in grid for cell in row)
                    obj_dict['shape'] = [['x' if cell != '.' else '.' for cell in row] for row in obj_dict['grid']]

                objects.append(obj_dict)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if not visited[i][j] and grid[i][j] == '.':
                obj = []
                dfs(i, j, grid[i][j], obj)
                row_bounds = (min(x for x, _ in obj), max(x for x, _ in obj))
                col_bounds = (min(y for _, y in obj), max(y for _, y in obj))
                obj_grid = [row[col_bounds[0]:col_bounds[1]+1] for row in grid[row_bounds[0]:row_bounds[1]+1]]
                obj_grid = [['$' for _ in range(col_bounds[1]-col_bounds[0]+1)] for _ in range(row_bounds[1]-row_bounds[0]+1)]
                obj_dict = {'tl': (i, j), 'grid': obj_grid}
                if more_info:
                    obj_dict['size'] = (len(grid), len(grid[0]))
                    obj_dict['cell_count'] = sum(cell != '.' for row in grid for cell in row)
                    obj_dict['shape'] = [['x' if cell != '.' else '.' for cell in row] for row in obj_dict['grid']]

                objects.append(obj_dict)

    return objects


# get_pixel_coords(grid):
#   Returns a dictionary, with the keys the pixel values, values the list of coords,
#   in sorted order from most number of pixels to least
def get_pixel_coords(array2d):
    pixel_view = {}
    for row_index, row in enumerate(array2d):
        for col_index, char in enumerate(row):
            if char not in pixel_view:
                pixel_view[char] = []
            pixel_view[char].append((row_index, col_index))
    return pixel_view

# empty_grid(row, col): returns an empty grid of height row and width col
def empty_grid(row, col):
    # No code needs to be inserted as the function is already implemented correctly
    return [['.' for _ in range(col)] for _ in range(row)]

# crop_grid(grid, tl, br): returns cropped section from top left to bottom right of the grid
def crop_grid(grid, tl, br):
    cropped_grid = [row[tl[1]:br[1]+1] for row in grid[tl[0]:br[0]+1]]
    return cropped_grid

# tight_fit(grid): returns grid with all blank rows and columns removed
def tight_fit(grid):
    # Remove empty rows
    grid = [row for row in grid if any(cell != '.' for cell in row)]

    # Transpose the grid for column operations
    grid_t = list(map(list, zip(*grid)))

    # Remove empty columns
    grid_t = [col for col in grid_t if any(cell != '.' for cell in col)]

    # Transpose back to original orientation
    grid = list(map(list, zip(*grid_t)))

    return grid

# combine_object(obj_1, obj_2): returns combined object from obj_1 and obj_2. if overlap, obj_2 overwrites obj_1
def combine_object(obj_1, obj_2):
    # Copy the grid of obj_1 to avoid modifying the original object
    combined_grid = [row.copy() for row in obj_1['grid']]

    # Calculate the relative position of obj_2 to obj_1
    relative_pos = (obj_2['tl'][0] - obj_1['tl'][0], obj_2['tl'][1] - obj_1['tl'][1])

    # Overwrite the cells in combined_grid with the cells of obj_2
    for i, row in enumerate(obj_2['grid']):
        for j, cell in enumerate(row):
            combined_grid[i + relative_pos[0]][j + relative_pos[1]] = cell

    # Return the combined object
    # Compute additional fields if they exist in the input objects
    additional_fields = {}
    if 'size' in obj_1 or 'size' in obj_2:
        additional_fields['size'] = (len(combined_grid), len(combined_grid[0]))
    if 'cell_count' in obj_1 or 'cell_count' in obj_2:
        additional_fields['cell_count'] = sum(cell != '.' for row in combined_grid for cell in row)
    if 'shape' in obj_1 or 'shape' in obj_2:
        additional_fields['shape'] = [len(row) for row in combined_grid if any(cell != '.' for cell in row)]

    # Combine the base object with any additional fields
    combined_object = {'tl': obj_1['tl'], 'grid': combined_grid}
    combined_object.update(additional_fields)
    return combine_object

# rotate_clockwise(grid, degree=90): returns rotated grid clockwise by a degree of 90, 180, 270 degrees
def rotate_clockwise(grid, degree=90):
    if degree == 90:
        return [list(reversed(col)) for col in zip(*grid)]
    elif degree == 180:
        return [list(reversed(row)) for row in reversed(grid)]
    elif degree == 270:
        return [list(col) for col in reversed(list(zip(*grid)))]
    else:
        return grid

# horizontal_flip(grid): returns a horizontal flip of the grid
def horizontal_flip(grid):
    return [row[::-1] for row in grid]

# vertical_flip(grid): returns a vertical flip of the grid
def vertical_flip(grid):
    return grid[::-1]

# replace(grid, grid_1, grid_2): replaces all occurences of grid_1 with grid_2 in grid
def replace(grid, grid_1, grid_2):
    for i in range(len(grid) - len(grid_1) + 1):
        for j in range(len(grid[0]) - len(grid_1[0]) + 1):
            if all(grid[i + x][j + y] == grid_1[x][y] for x in range(len(grid_1)) for y in range(len(grid_1[0]))):
                for x in range(len(grid_2)):
                    for y in range(len(grid_2[0])):
                        grid[i + x][j + y] = grid_2[x][y]

    return grid

# get_object_color(obj): returns color of object. if muloticolor, returns first color only
def get_object_color(obj):
    # Iterate over the grid of the object
    for row in obj['grid']:
        for cell in row:
            # If the cell is not empty, return its color
            if cell != '.':
                return cell
    # If the object is empty, return None
    return None


# change_object_color(obj, value): changes the object color to value
def change_object_color(obj, value):
    # Create a new grid with the new color
    new_grid = [[value if cell != '.' else '.' for cell in row] for row in obj['grid']]

    # Return the combined object
    # Compute additional fields if they exist in the input objects
    additional_fields = {}
    if 'size' in obj:
        additional_fields['size'] = obj["size"]
    if 'cell_count' in obj:
        additional_fields['cell_count'] = sum(cell != '.' for row in new_grid for cell in row)
    if 'shape' in obj:
        additional_fields['shape'] = [len(row) for row in new_grid if any(cell != '.' for cell in row)]

    # Combine the base object with any additional fields
    new_obj = {'tl': obj['tl'], 'grid': new_grid}
    new_obj.update(additional_fields)

    return new_obj

# fill_object(grid, obj, align=False): fills grid with object. If align is True, makes grid same size as object
def fill_object(grid, obj, align=False):
    # If align is True, make the grid the same size as the object
    if align:
        grid = [['.' for _ in range(len(obj['grid'][0]))] for _ in range(len(obj['grid']))]

    # Get the top left coordinates of the object
    tl_row, tl_col = obj['tl']

    # Iterate over the object's grid
    for i in range(len(obj['grid'])):
        for j in range(len(obj['grid'][0])):
            # If the cell in the object's grid is not empty, fill the corresponding cell in the grid
            if obj['grid'][i][j] != '.':
                grid[tl_row + i][tl_col + j] = obj['grid'][i][j]

    return grid

# fill_row(grid, row_num, value, start_col=0, end_col=30): fills output grid with a row of value at row_num from start_col to end_col (inclusive)
def fill_row(grid, row_num, value, start_col=0, end_col=30):
    # Check if row_num is within the grid
    if row_num < 0 or row_num >= len(grid):
        raise ValueError("row_num is out of grid bounds")

    # Check if start_col and end_col are within the grid
    if start_col < 0:
        print("Warning: start_col is out of grid bounds. It has been set to 0.")
        start_col = 0
    if end_col >= len(grid[0]):
        print("Warning: end_col is out of grid bounds. It has been set to the maximum column index.")
        end_col = len(grid[0]) - 1

    # Fill the row from start_col to end_col with value
    for col in range(start_col, end_col + 1):
        grid[row_num][col] = value

    return grid

# fill_col(grid, cool_num, value, start_row=0, end_row=30): fills output grid with a column of value at col_num from start_row to end_row (inclusive)
def fill_col(grid, col_num, value, start_row=0, end_row=30):
    # Check if col_num is within the grid
    if col_num < 0 or col_num >= len(grid[0]):
        raise ValueError("col_num is out of grid bounds")

    # Check if start_row and end_row are within the grid
    if start_row < 0:
        print("Warning: start_row is out of grid bounds. It has been set to 0.")
        start_row = 0
    if end_row >= len(grid):
        print("Warning: end_row is out of grid bounds. It has been set to the maximum row index.")
        end_row = len(grid) - 1

    # Fill the column from start_row to end_row with value
    for row in range(start_row, end_row + 1):
        grid[row][col_num] = value

    return grid

# fill_between_coords(grid, coord_1, coord_2, value): fills line between coord_1 and coord_2 with value
def fill_between_coords(grid, coord_1, coord_2, value):
    # Check if coord_1 and coord_2 are within the grid
    if coord_1[0] < 0 or coord_1[0] >= len(grid) or coord_1[1] < 0 or coord_1[1] >= len(grid[0]) or \
       coord_2[0] < 0 or coord_2[0] >= len(grid) or coord_2[1] < 0 or coord_2[1] >= len(grid[0]):
        raise ValueError("coord_1 or coord_2 is out of grid bounds")

    # Check if coord_1 and coord_2 are on the same line
    if coord_1[0] == coord_2[0]:  # Same row
        start_col = min(coord_1[1], coord_2[1])
        end_col = max(coord_1[1], coord_2[1])
        for col in range(start_col, end_col + 1):
            grid[coord_1[0]][col] = value
    elif coord_1[1] == coord_2[1]:  # Same column
        start_row = min(coord_1[0], coord_2[0])
        end_row = max(coord_1[0], coord_2[0])
        for row in range(start_row, end_row + 1):
            grid[row][coord_1[1]] = value
    else:  # Diagonal
        if coord_1[0] < coord_2[0]:
            if coord_1[1] < coord_2[1]:  # Diagonal from top left to bottom right
                for i in range(coord_2[0] - coord_1[0] + 1):
                    grid[coord_1[0] + i][coord_1[1] + i] = value
            else:  # Diagonal from top right to bottom left
                for i in range(coord_2[0] - coord_1[0] + 1):
                    grid[coord_1[0] + i][coord_1[1] - i] = value
        else:
            if coord_1[1] < coord_2[1]:  # Diagonal from bottom left to top right
                for i in range(coord_1[0] - coord_2[0] + 1):
                    grid[coord_1[0] - i][coord_1[1] + i] = value
            else:  # Diagonal from bottom right to top left
                for i in range(coord_1[0] - coord_2[0] + 1):
                    grid[coord_1[0] - i][coord_1[1] - i] = value

    return grid


# fill_rect(grid,tl,br,value): fills grid from tl to br with value. useful to create rows, columns, rectangles
def fill_rect(grid, tl, br, value):
    # Check if tl and br are within the grid
    if tl[0] < 0 or tl[0] >= len(grid) or tl[1] < 0 or tl[1] >= len(grid[0]) or \
       br[0] < 0 or br[0] >= len(grid) or br[1] < 0 or br[1] >= len(grid[0]):
        raise ValueError("tl or br is out of grid bounds")

    # Fill the rectangle from tl to br with value
    for row in range(tl[0], br[0] + 1):
        for col in range(tl[1], br[1] + 1):
            grid[row][col] = value

    return grid

# fill_value(grid, pos, value): fills grid at position with value
def fill_value(grid, pos, value):
    # Check if pos is within the grid
    if pos[0] < 0 or pos[1] >= len(grid) or pos[0] < 0 or pos[0] >= len(grid[0]):
        raise ValueError("pos {} is out of grid bounds".format(pos))

    # Fill the grid at pos with value
    grid[pos[0]][pos[1]] = value

    return grid



# object_contains_color(obj, value): returns True/False if object contains a certain value
def object_contains_color(obj, value):
    for row in obj['grid']:
        if value in row:
            return True
    return False


# on_same_line(coord_1, coord_2): Returns True/False if coord_1 is on the same line as coord_2. line_type can be one
# of [’row’, ’col’, ’diag’]
def on_same_line(coord_1, coord_2, line_type):
    if line_type == 'row':
        return coord_1[0] == coord_2[0]
    elif line_type == 'col':
        return coord_1[1] == coord_2[1]
    elif line_type == 'diag':
        return abs(coord_1[0] - coord_2[0]) == abs(coord_1[1] - coord_2[1])
    else:
        raise ValueError("Invalid line_type")

