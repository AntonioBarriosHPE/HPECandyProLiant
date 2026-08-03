# No installation needed, tkinter is a standard Python library

import tkinter

def get_screen_resolution():
    """Gets the primary screen's resolution."""
    try:
        # Create a tkinter root window
        root = tkinter.Tk()

        # Withdraw the window so it doesn't appear on the screen
        root.withdraw()

        # Get screen width and height
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        # Destroy the window
        root.destroy()

        return width, height
    except Exception as e:
        print(f"Could not get screen resolution: {e}")
        return None, None

if __name__ == "__main__":
    screen_width, screen_height = get_screen_resolution()

    if screen_width and screen_height:
        print(f"Your screen resolution is: {screen_width}x{screen_height}")