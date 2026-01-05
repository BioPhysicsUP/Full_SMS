"""Full SMS application entry point."""

import dearpygui.dearpygui as dpg


def main() -> None:
    """Run the Full SMS application."""
    dpg.create_context()
    dpg.create_viewport(title="Full SMS", width=1280, height=800)

    with dpg.window(label="Full SMS", tag="primary_window"):
        dpg.add_text("Full SMS - Single Molecule Spectroscopy Analysis")
        dpg.add_text("Application is loading...")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
