import matplotlib.pyplot as plt


def plot_3d_coordinates(coordinates):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = [coord[0] for coord in coordinates]
    ys = [coord[1] for coord in coordinates]
    zs = [coord[2] for coord in coordinates]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def read_bytes_and_create_3d_coordinates(file_path):
    with open(file_path, 'rb') as file:
        bytes_data = file.read()

    return create_3d_coordinates_from_bytes(bytes_data)


def create_3d_coordinates_from_bytes(bytes_data):
    coordinates = []
    for i in range(0, len(bytes_data) - (len(bytes_data) % 3), 3):
        x = bytes_data[i]
        y = bytes_data[i + 1]
        z = bytes_data[i + 2]
        coordinates.append((x, y, z))
    return coordinates


file_path = '/usr/bin/cat'
coordinates = plot_3d_coordinates(read_bytes_and_create_3d_coordinates(file_path))
