import napari

v = napari.Viewer()
sl = v.add_shapes(scale=[100, 100])
sl.add_paths([[0, 0], [1, 1], [3, 4]])
sl.selected_data = {0}
sl.mode = 'select'

# playin around
shape2 = v.add_shapes(scale=[100, 100])
shape2.add_paths([[2, 2], [3,3]])
shape2.selected_data = {0}
shape2.mode = 'select'

shape3 = v.add_shapes(scale=[100, 100])
shape3.add_paths([[4, 4], [5,5]])
shape3.selected_data = {0}
shape3.mode = 'select'

napari.run()
