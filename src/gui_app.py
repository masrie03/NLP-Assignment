"""
gui_app.py

Graphical User Interface for the arXiv-based Spelling Correction System.

This module ONLY handles presentation and user interaction. All linguistic
computation (dictionary lookup, Levenshtein distance, bigram smoothing) is
delegated to nlp_engine.SpellCheckEngine, which loads the pre-built pickle
files instantly at startup (no re-processing of the raw corpus is ever done
here).

Layout:
    +--------------------------------------------+------------------------+
    |  Text editor (max 500 characters)           |  Dictionary Explorer   |
    |  [Check Spelling]                           |  [search box]          |
    |                                              |  [word list + freq]   |
    +--------------------------------------------+------------------------+

Run with:
    python gui_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

from nlp_engine import SpellCheckEngine, DetectedError


MAX_CHARACTERS = 500

COLOR_NONWORD = "#ff8080"      # red highlight
COLOR_REALWORD = "#ffe066"     # yellow highlight

NAVIGATION_KEYS = {
    "Left", "Right", "Up", "Down", "Home", "End",
    "Prior", "Next", "Shift_L", "Shift_R", "Control_L", "Control_R",
    "BackSpace", "Delete", "Tab",
}


class SuggestionPopup(tk.Toplevel):
    """Small pop-up window shown when the user clicks a highlighted word.

    Displays the error type, a ranked list of suggestions, and an
    "Apply Correction" button that replaces the word in the editor and
    triggers a fresh spell-check pass.
    """

    def __init__(self, master, error: DetectedError, on_apply):
        super().__init__(master)
        self.title("Correction suggestions")
        self.resizable(False, False)
        self.on_apply = on_apply
        self.error = error

        label_type = "Non-word error" if error.error_type == "nonword" else "Real-word (contextual) error"
        tk.Label(self, text=f'"{error.word}"  —  {label_type}',
                  font=("Segoe UI", 10, "bold")).pack(padx=12, pady=(12, 4))

        self.listbox = tk.Listbox(self, height=min(6, max(1, len(error.suggestions))), width=30)
        for suggestion in error.suggestions:
            self.listbox.insert(tk.END, suggestion)
        if error.suggestions:
            self.listbox.selection_set(0)
        else:
            self.listbox.insert(tk.END, "(no suggestion found)")
        self.listbox.pack(padx=12, pady=4)

        button_frame = tk.Frame(self)
        button_frame.pack(pady=(4, 12))
        tk.Button(button_frame, text="Apply Correction",
                  command=self._apply).pack(side=tk.LEFT, padx=4)
        tk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _apply(self):
        if not self.error.suggestions:
            self.destroy()
            return
        selection = self.listbox.curselection()
        chosen_word = self.listbox.get(selection[0]) if selection else self.error.suggestions[0]
        self.on_apply(self.error, chosen_word)
        self.destroy()


class SpellCheckApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("arXiv Spelling Correction System")
        self.geometry("1000x600")

        # Load the pre-built dictionaries once at startup.
        self.engine = SpellCheckEngine()

        # error_id -> DetectedError, used to look up which word was clicked
        self._active_errors: dict[str, DetectedError] = {}

        self._build_layout()
        self._populate_dictionary_explorer("")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6)
        main_pane.pack(fill=tk.BOTH, expand=True)

        self._build_editor_panel(main_pane)
        self._build_explorer_panel(main_pane)

    def _build_editor_panel(self, parent):
        editor_frame = tk.Frame(parent, padx=10, pady=10)
        parent.add(editor_frame, stretch="always")

        top_bar = tk.Frame(editor_frame)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text="Text Editor", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        self.char_count_label = tk.Label(top_bar, text=f"0 / {MAX_CHARACTERS}")
        self.char_count_label.pack(side=tk.RIGHT)

        self.text_editor = tk.Text(editor_frame, wrap=tk.WORD, font=("Segoe UI", 12),
                                    undo=True, height=20)
        self.text_editor.pack(fill=tk.BOTH, expand=True, pady=(6, 6))
        self.text_editor.tag_configure("nonword", background=COLOR_NONWORD)
        self.text_editor.tag_configure("realword", background=COLOR_REALWORD)

        self.text_editor.bind("<KeyPress>", self._enforce_character_limit)
        self.text_editor.bind("<KeyRelease>", self._update_char_counter)

        bottom_bar = tk.Frame(editor_frame)
        bottom_bar.pack(fill=tk.X)
        tk.Button(bottom_bar, text="Check Spelling",
                  command=self.run_spell_check).pack(side=tk.LEFT)

        legend = tk.Frame(bottom_bar)
        legend.pack(side=tk.RIGHT)
        tk.Label(legend, text="  Non-word  ", bg=COLOR_NONWORD).pack(side=tk.LEFT, padx=4)
        tk.Label(legend, text="  Real-word  ", bg=COLOR_REALWORD).pack(side=tk.LEFT, padx=4)

    def _build_explorer_panel(self, parent):
        explorer_frame = tk.Frame(parent, padx=10, pady=10, width=280)
        parent.add(explorer_frame)

        tk.Label(explorer_frame, text="Dictionary Explorer",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(explorer_frame,
                  text=f"{self.engine.vocabulary_size:,} unique words (arXiv corpus)",
                  fg="gray").pack(anchor="w", pady=(0, 6))

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(explorer_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X)
        self.search_var.trace_add("write", self._on_search_changed)

        list_frame = tk.Frame(explorer_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.dictionary_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.dictionary_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.dictionary_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Text editor: 500-character hard limit + live counter
    # ------------------------------------------------------------------
    def _current_length(self) -> int:
        # "-1c" strips the trailing newline that Tkinter Text always adds.
        return len(self.text_editor.get("1.0", "end-1c"))

    def _enforce_character_limit(self, event):
        if event.keysym in NAVIGATION_KEYS:
            return None
        # Allow the keystroke only if we are still under the limit, or if
        # there is an active selection (which will be replaced, not added).
        has_selection = bool(self.text_editor.tag_ranges("sel"))
        if self._current_length() >= MAX_CHARACTERS and not has_selection:
            return "break"
        return None

    def _update_char_counter(self, event=None):
        length = self._current_length()
        self.char_count_label.config(text=f"{length} / {MAX_CHARACTERS}")

    # ------------------------------------------------------------------
    # Spell-checking + highlighting
    # ------------------------------------------------------------------
    def run_spell_check(self):
        text = self.text_editor.get("1.0", "end-1c")

        # Clear previous highlights and click bindings.
        self.text_editor.tag_remove("nonword", "1.0", tk.END)
        self.text_editor.tag_remove("realword", "1.0", tk.END)
        self._active_errors.clear()

        errors = self.engine.check_text(text)

        for i, error in enumerate(errors):
            start_index = f"1.0 + {error.start} chars"
            end_index = f"1.0 + {error.end} chars"

            self.text_editor.tag_add(error.error_type, start_index, end_index)

            error_tag = f"err_{i}"
            self._active_errors[error_tag] = error
            self.text_editor.tag_add(error_tag, start_index, end_index)
            self.text_editor.tag_bind(
                error_tag, "<Button-1>",
                lambda event, tag=error_tag: self._on_error_click(tag)
            )

        if not errors:
            messagebox.showinfo("Spell Check", "No errors detected.")

    def _on_error_click(self, error_tag: str):
        error = self._active_errors.get(error_tag)
        if error is None:
            return
        SuggestionPopup(self, error, on_apply=self._apply_correction)

    def _apply_correction(self, error: DetectedError, chosen_word: str):
        start_index = f"1.0 + {error.start} chars"
        end_index = f"1.0 + {error.end} chars"
        self.text_editor.delete(start_index, end_index)
        self.text_editor.insert(start_index, chosen_word)
        self._update_char_counter()
        # Re-run the check automatically, as required by the specification.
        self.run_spell_check()

    # ------------------------------------------------------------------
    # Dictionary explorer (live search)
    # ------------------------------------------------------------------
    def _on_search_changed(self, *_args):
        self._populate_dictionary_explorer(self.search_var.get())

    def _populate_dictionary_explorer(self, prefix: str):
        self.dictionary_listbox.delete(0, tk.END)
        results = self.engine.search_dictionary(prefix, limit=300)
        for word, freq in results:
            self.dictionary_listbox.insert(tk.END, f"{word}   ({freq})")


def main():
    app = SpellCheckApp()
    app.mainloop()


if __name__ == "__main__":
    main()
