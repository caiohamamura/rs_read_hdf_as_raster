use ndarray::{s, Array, SliceInfo};
use std::iter::FromIterator;
use std::{io, io::Write};

pub trait HasMembers {
    fn get_members(&self) -> Result<Vec<String>, hdf5::Error>;
    fn get_group(&self, name: &str) -> Result<hdf5::Group, hdf5::Error>;
    fn is_group(&self, name: &str) -> bool;
}

#[derive(Debug)]
enum H5NodeType {
    Dataset(String),
    Group(String),
}

impl HasMembers for hdf5::File {
    fn get_members(&self) -> Result<Vec<String>, hdf5::Error> {
        return self.member_names();
    }

    fn get_group(&self, name: &str) -> Result<hdf5::Group, hdf5::Error> {
        return self.group(name);
    }

    fn is_group(&self, name: &str) -> bool {
        return self.link_exists(name);
    }
}

impl HasMembers for hdf5::Group {
    fn get_members(&self) -> Result<Vec<String>, hdf5::Error> {
        return self.member_names();
    }

    fn get_group(&self, name: &str) -> Result<hdf5::Group, hdf5::Error> {
        return self.group(name);
    }

    fn is_group(&self, name: &str) -> bool {
        return self.link_exists(name);
    }
}

fn ls_hdf5(obj: &impl HasMembers, parent: String) -> Vec<H5NodeType> {
    let mut result: Vec<H5NodeType> = vec![];
    if let Ok(member_names) = obj.get_members() {
        for member_name in member_names {
            let new_parent = parent.clone() + "/" + member_name.as_str();
            {
                let _silence = hdf5::silence_errors();
                if let Ok(group) = obj.get_group(member_name.as_str()) {
                    result.push(H5NodeType::Group(new_parent.clone()));
                    result.append(&mut ls_hdf5(&group, new_parent))
                } else {
                    result.push(H5NodeType::Dataset(new_parent));
                }
            }
        }
    }
    return result;
}

fn rev_array<T: Clone>(
    input: ndarray::Array<T, ndarray::Dim<[usize; 1]>>,
    nrows: usize,
    ncols: usize,
) -> ndarray::Array<T, ndarray::Dim<[usize; 1]>> {
    let input = input.into_shape((nrows, ncols)).unwrap();
    let input = input.slice(s![..;-1, ..]);
    let input = Array::from_iter(input.iter().cloned());
    return input;
}

fn reverse_ds_rows<T: hdf5::H5Type + Clone>(
    file: &hdf5::File,
    base_ds: String,
    xsize: usize,
    ysize: usize,
) {
    if base_ds.ends_with("_rev") {
        return;
    }
    let ds_name_rev = base_ds.clone() + "_rev";
    if file.link_exists(&ds_name_rev) {
        return;
    }
    let ds: hdf5::Dataset = file.dataset(&base_ds).unwrap();
    let ds_out = create_dataset::<T>(&file, &ds_name_rev, ds.size());

    let half_lines = (ysize as f32 / 2f32).ceil() as usize;
    let n_lines_read = 100usize;

    for yy in (0..half_lines).step_by(n_lines_read) {
        let perc = 100f32 * yy as f32 / (half_lines) as f32;
        print!("\r{:.2}%", perc);
        io::stdout().flush().unwrap();

        let mut lines_to_read = n_lines_read;

        if (yy + n_lines_read) > half_lines {
            lines_to_read = half_lines - yy;
        }

        let rev_yy = ysize - yy - 1 - lines_to_read;
        let lower_bound = yy * xsize;
        let upper_bound = yy * xsize + lines_to_read * xsize;
        let rev_lower_bound = rev_yy * xsize;
        let rev_upper_bound = rev_yy * xsize + lines_to_read * xsize;
        let slice_or_info = s![lower_bound..upper_bound];
        let slice = SliceInfo::new(slice_or_info).unwrap();
        let rev_slice_or_info = s![rev_lower_bound..rev_upper_bound];
        let rev_slice = SliceInfo::new(rev_slice_or_info).unwrap();

        let vals = ds.read_slice_1d::<T, _>(&slice).unwrap();
        let vals_final = rev_array(vals, lines_to_read, xsize);

        let rev_vals = ds.read_slice_1d::<T, _>(&rev_slice).unwrap().clone();
        let rev_vals_final = rev_array(rev_vals, lines_to_read, xsize);

        ds_out
            .write_slice(rev_vals_final.as_slice().unwrap(), &slice)
            .unwrap();
        ds_out
            .write_slice(vals_final.as_slice().unwrap(), &rev_slice)
            .unwrap();
    }
    println!("\r{:.2}%", 100f32);
}

fn create_dataset<T: hdf5::H5Type>(file: &hdf5::File, name: &str, size: usize) -> hdf5::Dataset {
    let mut ds_builder = file.new_dataset::<T>();
    ds_builder.gzip(1);

    let ds_out = ds_builder.create(name, size).unwrap();
    return ds_out;
}

fn calc_mean_sd(file: &hdf5::File, group_name: &str, chunk_size: usize) {
    let sum_path = String::from("/") + group_name + "/sum_rev";
    let sumsq_path = String::from("/") + group_name + "/sumsq_rev";
    let count_path = String::from("/") + group_name + "/count_rev";
    let mean_path_out = String::from("/") + group_name + "/mean_rev";
    let sd_path_out = String::from("/") + group_name + "/sd_rev";

    if file.link_exists(&mean_path_out) {
        return;
    }
    let sum_ds: hdf5::Dataset = file.dataset(&sum_path).unwrap();
    let sumsq_ds: hdf5::Dataset = file.dataset(&sumsq_path).unwrap();
    let count_ds: hdf5::Dataset = file.dataset(&count_path).unwrap();
    let max_size = sum_ds.size();

    let mean_ds_out: hdf5::Dataset = create_dataset::<f32>(&file, &mean_path_out, max_size);
    let sd_ds_out: hdf5::Dataset = create_dataset::<f32>(&file, &sd_path_out, max_size);

    for ii in (0..max_size).step_by(chunk_size) {
        let mut n_vals_read = chunk_size;

        if (ii + n_vals_read) > max_size {
            n_vals_read = max_size - ii;
        }

        let slice = s![ii..(ii + n_vals_read)];
        let the_slice = SliceInfo::new(slice).unwrap();

        let sum_vals = sum_ds.read_slice_1d::<f32, _>(&the_slice).unwrap();
        let sumsq_vals = sumsq_ds.read_slice_1d::<f32, _>(&the_slice).unwrap();
        let count_vals = count_ds.read_slice_1d::<u8, _>(&the_slice).unwrap();
        
        let mask = count_vals.mapv(|el| el == 0);
        let count_vals = count_vals.mapv(|el| el as f32);

        let mean = &sum_vals / &count_vals;
        let variance =
            (sumsq_vals - (sum_vals.mapv(|el| el.powi(2)) / &count_vals)) / (&count_vals - 1f32);
        let mut sd = variance.mapv(|el| el.sqrt() );

        ndarray::Zip::from(&mut sd).and(&mask).apply(|x, &m| if m {
            *x = -1f32;
        });

        let _ = mean_ds_out.write_slice(&mean, &the_slice);
        let _ = sd_ds_out.write_slice(&sd, &the_slice);
    }
}

fn main() {
    let file: hdf5::file::File = hdf5::file::File::open_rw("cerrado_100.h5").unwrap();
    let base_float_path = "base_float.tif";
    let base_byte_path = "base_byte.tif";
    let base_rast = gdal::Dataset::open(std::path::Path::new(base_byte_path)).unwrap();
    let base_band = base_rast.rasterband(1).unwrap();
    let xsize = base_band.x_size();
    let ysize = base_band.y_size();

    let hdf5_nodes: Vec<H5NodeType> = ls_hdf5(&file, "".to_owned());

    let datasets: Vec<&H5NodeType> = hdf5_nodes
        .iter()
        .filter(|&x| match x {
            H5NodeType::Dataset(_) => true,
            _ => false,
        })
        .collect();

    let groups: Vec<&H5NodeType> = hdf5_nodes
        .iter()
        .filter(|&x| match x {
            H5NodeType::Group(_) => true,
            _ => false,
        })
        .collect();

    let total_datasets = datasets.len();
    let mut counter = 0;

    println!("Inverting datasets rows!");
    datasets.iter().for_each(|ds| {
        if let H5NodeType::Dataset(ds_name) = ds {
            counter += 1;
            println!("Processing dataset: {} ({} of {})", ds_name, counter, total_datasets);
            if ds_name.ends_with("count") {
                reverse_ds_rows::<u8>(&file, ds_name.to_string(), xsize, ysize);
            } else {
                reverse_ds_rows::<f32>(&file, ds_name.to_string(), xsize, ysize);
            }
        };
    });

    let chunk_size = 1000000;
    let total_groups = groups.len();
    let mut counter = 0;
    groups.iter().for_each(|group| {
        counter += 1;
        if let H5NodeType::Group(group_name) = group {
            println!("Processing group: {} ({} of {})", group_name, counter, total_groups);
            println!("Computing mean and sd...");
            calc_mean_sd(&file, group_name, chunk_size);
            println!("Finished!");
            let group_name = group_name.replace("/", "");
            
            let ds_count_path = format!("/{}/count_rev", group_name);
            let ds_mean_path = format!("/{}/mean_rev", group_name);
            let ds_sd_path = format!("/{}/sd_rev", group_name);

            let ds_count = file.dataset(&ds_count_path).unwrap();
            let ds_mean = file.dataset(&ds_mean_path).unwrap();
            let ds_sd = file.dataset(&ds_sd_path).unwrap();

            // let the_slice = s![(1219+1088*2137)..(1230+1088*2137)];
            // let the_slice_info = SliceInfo::new(the_slice).unwrap();
            let out_mean_path = format!("{}_cerrado_{}_{}.tif", 100, group_name, "mean");
            std::fs::copy(base_float_path, &out_mean_path).unwrap();
            let rast_mean = gdal::Dataset::open_ex(
                std::path::Path::new(&out_mean_path),
                Some(gdal_sys::GDALAccess::GA_Update),
                None,
                None,
                None,
            )
            .unwrap();
            let band_mean = rast_mean.rasterband(1).unwrap();

            let out_sd_path = format!("{}_cerrado_{}_{}.tif", 100, group_name, "sd");
            std::fs::copy(base_float_path, &out_sd_path).unwrap();
            let rast_sd = gdal::Dataset::open_ex(
                std::path::Path::new(&out_sd_path),
                Some(gdal_sys::GDALAccess::GA_Update),
                None,
                None,
                None,
            )
            .unwrap();
            let band_sd = rast_sd.rasterband(1).unwrap();

            let out_count_path = format!("{}_cerrado_{}_{}.tif", 100, group_name, "count");
            std::fs::copy(base_byte_path, &out_count_path).unwrap();
            let rast_count = gdal::Dataset::open_ex(
                std::path::Path::new(&out_count_path),
                Some(gdal_sys::GDALAccess::GA_Update),
                None,
                None,
                None,
            )
            .unwrap();
            let band_count = rast_count.rasterband(1).unwrap();

            let n_lines_read = 100;
            println!("Reading HDF and writing to rasters...");
            for yy in (0..ysize).step_by(n_lines_read) {
                let perc = 100f32 * yy as f32 / ysize as f32;
                if perc.round() as u32 % 2 == 0 {
                    print!("\r{:.2}%", perc);
                    io::stdout().flush().unwrap();
                }
                let mut lines_to_read = n_lines_read;
                if (yy + n_lines_read) > ysize {
                    lines_to_read = ysize - yy;
                }

                let lower_bound = yy * xsize;
                let upper_bound = yy * xsize + lines_to_read * xsize;
                let slice_or_info = s![lower_bound..upper_bound];
                let slice = SliceInfo::new(slice_or_info).unwrap();
                let count = ds_count.read_slice::<u8, _, _>(&slice).unwrap();
                let mean = ds_mean.read_slice::<f32, _, _>(&slice).unwrap();
                let sd = ds_sd.read_slice::<f32, _, _>(&slice).unwrap();

                let buffer_count =
                    gdal::raster::Buffer::<u8>::new((xsize, lines_to_read), count.to_vec());

                let buffer_mean =
                    gdal::raster::Buffer::<f32>::new((xsize, lines_to_read), mean.to_vec());

                let buffer_sd =
                    gdal::raster::Buffer::<f32>::new((xsize, lines_to_read), sd.to_vec());

                band_count.write((0, yy as isize), (xsize, lines_to_read), &buffer_count).unwrap();
                band_mean.write((0, yy as isize), (xsize, lines_to_read), &buffer_mean).unwrap();
                band_sd.write((0, yy as isize), (xsize, lines_to_read), &buffer_sd).unwrap();
            }
            println!("\r{:.2}%", 100f32);
            println!("Finished!");
        }
    });
}
