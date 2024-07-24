Examples
=========

This folder contains different examples for the user to run with the provided data. Here is the list on how to use the different configuration files.

Video Generation
-------------------

This will generate a video for the given value of force on the entry `video_generator/force` on the configuration json file. The video will be saved at `data/results`. To run this example simply run:

**Liver video**

```shell
python pymodal_surgical.apps.video_generator --config examples/liver_video_generator.json 
```

**Heart beating video**

```shell
python pymodal_surgical.apps.video_generator --config examples/heart_beating_generator.json
```

