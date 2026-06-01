import customtkinter as ctk
from ui.app_window import AppWindow


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = AppWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
