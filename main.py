from nicegui import ui

from convert import create_coordinates_from_bytes, create_image


current_image = None
remove_button = None


def handle_upload(e):
    global current_image, remove_button

    ui.notify(f"Uploaded {e.name}")

    text = e.content.read()
    coordinates = create_coordinates_from_bytes(text)

    image_data = create_image(coordinates, tuple([256, 256]))

    if current_image:
        current_image.delete()
    if remove_button:
        remove_button.delete()

    current_image = ui.image(image_data)
    remove_button = ui.button("Remove", on_click=lambda e: remove_image_and_button())


def remove_image_and_button():
    global current_image, remove_button
    if current_image:
        current_image.delete()
        current_image = None
    if remove_button:
        remove_button.delete()
        remove_button = None


with ui.label("B2V"):
    ui.upload(on_upload=handle_upload).classes("max-w-full")

ui.run()
