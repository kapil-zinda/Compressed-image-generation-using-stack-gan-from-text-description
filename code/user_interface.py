import PySimpleGUI as sg
import os.path
from main import runner

# First the window layout in 2 columns

# text/001.Black_footed_Albatross/Black_footed_Albatross_0002_55
# text/136.Barn_Swallow/Barn_Swallow_0017_132951
# text/016.Painted_Bunting/Painted_Bunting_0001_16585
file_name = "/data/birds/example_captions.txt"
with open(file_name,"w"):
    pass
file_list_column = [
    [
        sg.Text("Text Discription --- "),
        sg.In(size=(25, 1), enable_events=True, key="-DIS-"),
        sg.Button("OK"),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "OK":
        txt_dis = values["-DIS-"]

        try:
            # Get list of files in folder
            file1 = open(file_name,"w")
            
            # \n is placed to indicate EOL (End of Line)
            file1.write(txt_dis)
            file1.close()
            runner()
            folder = "/models/bird_DMGAN/example_captions/"
            file_list = os.listdir(folder)
            break
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
        break
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-DIS-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass


while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "OK":
        txt_dis = values["-DIS-"]

        try:
            folder = "/models/bird_DMGAN/example_captions/"
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                folder, values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)

        except:
            pass

window.close()