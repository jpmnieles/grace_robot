# Grace Robot
Repository for Grace Robot Eye Controllers

## File Structure

~~~
/data
    /raw
        /EXP1_Name
            /DATE1_ID_Trial
                /CSV_Task1.csv
                /CSV_Task2.csv
                /PICKLE_All.pickle
                /metadata.json
            /230801_618082_Trial
                /CSV_Task1.csv
                /CSV_Task2.csv
                /PICKLE_All.pickle
                /metadata.json
            /230801_618082_Final
                /CSV_Task1.csv
                /CSV_Task2.csv
                /PICKLE_All.pickle
                /metadata.json
    /processed
        /EXP1_Name
            /DATE1_ID_Trial
                /CSV_Task1.csv
                /CSV_Task2.csv
                /metadata.json
            /230801_618082_Trial
                /CSV_Task1.csv
                /CSV_Task2.csv
                /metadata.json

/results
    /EXP1_Name1
        /DATE1_ID_Trial
            /IMG_1.jpg
            /IMG_2.jpg
            /metadata.json

~~~
* Trial = Exploration Data
* Final = Used for analysis


### Metadata
* Json File
* Contains the git hash of the script used
* Metadata is sent to the exported folder
~~~
{
    "exp_num": 1,
    "exp_name": "Task 1",
    "script": "script.py",
    "hash": "h2se234",
    "final": False,
    "date": "2023-08-15",
    "timestamp: "1212334343.1343434"
    "tasks": [
            "params": {
                "param1": 10,
                "param2": "abc",
                "param3": [1, 2, 3]},
            "params": {
                "param1": 10,
                "param2": "abc",
                "param3": [1, 2, 3]},
            ]
    "import": {
        [
            "params": {
                "filename": "CSV_Task.csv",
                "type: "raw",
                "exp_num": 1,
                "exp_name": "Task 1",
                "date": 20230801,
                "id": 618082,
                "final": False,
                "dir": None,  // If not None, use dir loc
            }
            "params": {
                "filename": "CSV_Task.csv",
                "type: "processed"
                "exp_num": 1,
                "exp_name": "Task 1",
                "date": 20230801,
                "id": 618082,
                "final": False,
                "dir": None,  // If not None, use dir loc
            }
        ]
    }
    "export": {
        [
            "params": {
                "filename": "CSV_Task.csv",
                "type: "raw",  // "raw", "processed", or "results"
                "exp_num": 1,
                "exp_name": "Task 1",
                "date": 20230801,
                "id": 618082,
                "final": False,
            }
            "params": {
                "filename": "IMG_1.jpg",
                "type: "results",  // "raw", "processed", or "results"
                "exp_num": 1,
                "exp_name": "Task 1",
                "date": 20230801,
                "id": 618082,
                "final": False,
            }
        ]
    }
}
~~~
