import trimesh
import os

def view_mesh(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load the mesh
    # trimesh automatically handles the .mtl file if it's in the same folder
    mesh = trimesh.load(file_path)

    print(f"Loading: {file_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Bounding Box: {mesh.bounds}") # Useful to check if scale is in meters or mm

    # Open the interactive viewer
    mesh.show()

if __name__ == "__main__":
    # Example: Viewing your base link
    # Adjust this path to match your structure
    mesh_to_open = "assets/meshes/base_link.obj"
    view_mesh(mesh_to_open)