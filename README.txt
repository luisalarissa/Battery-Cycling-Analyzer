BATTERY CYCLING ANALYZER
========================

A simple app to plot and analyze battery data from Arbin and PAT-Tester
instruments, so you don't have to do it by hand.


GET STARTED
-----------
1. Download this repository: green "Code" button > "Download ZIP",
   then unzip it.

2. Install the packages (only once). In Anaconda Prompt or Miniforge
   Prompt, go to the unzipped folder and run:

       pip install -r requirements.txt

3. Open battery_app.py in Spyder and press F5.


HOW TO USE IT
-------------
1. Click "Load files" and choose one of the example files (sample_Arbin_Data.xls or PAT_Cell_Impedance_example.txt).
2. In the table, type the active material mass in mg in the "Load" column.
3. Choose cycles on the right (for example: 1 2 3) and press "Apply".
4. Switch between plots with the tabs at the top: Profiles, Capacity & CE,
   Rate, dQ/dV, CV, Impedance, and more.
5. Export with the "Export" panel on the right: copy to Excel or Origin,
   or save a figure.


FILES IT READS
--------------
- Arbin:       .xls / .xlsx
- PAT-Tester:  .txt / .csv (including the impedance and C12 outputs)


LICENSE
-------
Free for noncommercial use (research, teaching, study) under the
PolyForm Noncommercial License 1.0.0. See the LICENSE file.
Commercial use needs permission from the author.

Made by Luisa Gomes-Popescu - https://github.com/luisalarissa
