import os, zipfile, yaml, shutil, random, uuid
from pathlib import Path

random.seed(42)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT = BASE_DIR / "campus_safety_v2"

for s in ['train','valid','test']:
    (OUT/s/'images').mkdir(parents=True, exist_ok=True)
    (OUT/s/'labels').mkdir(parents=True, exist_ok=True)

class_data = {0:[],1:[],2:[],3:[]}

def get_images_labels_from_zip(zp, tid, tname, keep_classes):
    print(f"\n=== {tname} (target ID: {tid}) ===")
    z = zipfile.ZipFile(zp, 'r')
    names = z.namelist()
    
    yaml_content = None
    for n in names:
        if n.endswith('data.yaml'):
            yaml_content = yaml.safe_load(z.read(n))
            break
    
    class_id_map = {}
    if yaml_content:
        orig = yaml_content.get('names', {})
        if isinstance(orig, dict):
            orig = list(orig.values())
        print(f"Classes: {orig}")
        
        for i,n in enumerate(orig):
            n_lower = str(n).lower()
            for keep in keep_classes:
                if keep.lower() in n_lower:
                    class_id_map[i] = keep
                    print(f"  KEEP: {n} -> {keep}")
                    break
    
    if not class_id_map:
        print(f"  WARNING: No matching classes found! Using all classes.")
        class_id_map = {i:str(i) for i in range(100)}
    
    files = [n for n in names if n.endswith(('.jpg','.png','.jpeg','.bmp'))]
    print(f"Images: {len(files)}")
    
    collected = 0
    for fpath in files:
        img_data = z.read(fpath)
        ext = Path(fpath).suffix
        
        lpath = fpath.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
        ldata = None
        try:
            ldata = z.read(lpath).decode('utf-8')
        except:
            pass
        
        if ldata:
            boxes = []
            for line in ldata.strip().split('\n'):
                p = line.split()
                if len(p)>=5:
                    cls = int(p[0])
                    if cls in class_id_map:
                        boxes.append(p[1:])
            
            if boxes:
                class_data[tid].append((img_data, ext, boxes))
                collected += 1
    
    z.close()
    print(f"Collected: {collected}")
    return collected

zips = [
    ("wet-floor-detection1.v2i.yolov8.zip",0,"wet_floor_sign",["Wet Floor"]),
    ("Fire Alarm.v24i.yolov8 (1).zip",1,"fire_alarm",["Fire alarm"]),
    ("Emergency Exit Signs.v4i.yolov8.zip",2,"emergency_exit",["Exit"]),
    ("Hard Hat Universe.v4i.yolov8.zip",3,"safety_helmet",["Helmet"]),
]

for zn,tid,tn, keep in zips:
    zp = BASE_DIR/zn
    if zp.exists():
        get_images_labels_from_zip(zp,tid,tn,keep)

print("\n=== SPLIT ===")
for c in class_data:
    random.shuffle(class_data[c])

for tid,tname in enumerate(['wet_floor_sign','fire_alarm','emergency_exit','safety_helmet']):
    p = class_data[tid]
    n = len(p)
    nt = int(0.7*n)
    nv = int(0.2*n)
    
    tr = p[:nt]
    va = p[nt:nt+nv]
    te = p[nt+nv:]
    
    print(f"{tname}: T={len(tr)} V={len(va)} Te={len(te)}")
    
    for sn,sp in [('train',tr),('valid',va),('test',te)]:
        for imgd,ext,bxs in sp:
            uuid_str = uuid.uuid4().hex[:8]
            iname = f"{tname}_{uuid_str}{ext}"
            lname = f"{tname}_{uuid_str}.txt"
            
            (OUT/sn/'images'/iname).write_bytes(imgd)
            with open(OUT/sn/'labels'/lname,'w') as f:
                for bx in bxs:
                    f.write(f"{tid} {' '.join(bx)}\n")

yaml.dump({
    'train': str(OUT/'train'/'images'),
    'val': str(OUT/'valid'/'images'),
    'test': str(OUT/'test'/'images'),
    'nc':4,
    'names':['wet_floor_sign','fire_alarm','emergency_exit','safety_helmet']
}, open(OUT/'data.yaml','w'))

print("\n=== DONE ===")
targets = {
    'wet_floor_sign': (480, 137, 69),
    'fire_alarm': (590, 170, 85),
    'emergency_exit': (900, 257, 128),
    'safety_helmet': (5000, 1400, 700)
}

for tn in ['wet_floor_sign','fire_alarm','emergency_exit','safety_helmet']:
    nt = len(list((OUT/'train'/'labels').glob(f"{tn}_*")))
    nv = len(list((OUT/'valid'/'labels').glob(f"{tn}_*")))
    nte = len(list((OUT/'test'/'labels').glob(f"{tn}_*")))
    mt, mv, mte = targets[tn]
    status = "OK" if (nt>=mt and nv>=mv and nte>=mte) else "LOW"
    print(f"{tn}: T={nt}({mt}) V={nv}({mv}) Te={nte}({mte}) [{status}]")

print(f"\nDataset: {OUT}")