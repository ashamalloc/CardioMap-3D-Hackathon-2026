-- CardioMap 3D · MySQL 8 schema
-- Usage:  mysql -u root -p < database/schema.sql

CREATE DATABASE IF NOT EXISTS cardiomap CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cardiomap;

-- Optional: dedicated app user (change the password and match backend/.env)
-- CREATE USER IF NOT EXISTS 'cardiomap_user'@'localhost' IDENTIFIED BY 'change_me';
-- GRANT ALL PRIVILEGES ON cardiomap.* TO 'cardiomap_user'@'localhost';
-- FLUSH PRIVILEGES;

-- Training dataset (final_scaled_data.csv). FBS, TG, WBC, PLT are stored as ln(raw), as in the CSV.
CREATE TABLE IF NOT EXISTS patient_records (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `age` SMALLINT UNSIGNED NOT NULL,
  `sex` TINYINT UNSIGNED NOT NULL,
  `bmi` DOUBLE NOT NULL,
  `dm` TINYINT UNSIGNED NOT NULL,
  `htn` TINYINT UNSIGNED NOT NULL,
  `current_smoker` TINYINT UNSIGNED NOT NULL,
  `ex_smoker` TINYINT UNSIGNED NOT NULL,
  `fh` TINYINT UNSIGNED NOT NULL,
  `crf` TINYINT UNSIGNED NOT NULL,
  `cva` TINYINT UNSIGNED NOT NULL,
  `airway_disease` TINYINT UNSIGNED NOT NULL,
  `thyroid_disease` TINYINT UNSIGNED NOT NULL,
  `chf` TINYINT UNSIGNED NOT NULL,
  `dlp` TINYINT UNSIGNED NOT NULL,
  `bp` SMALLINT UNSIGNED NOT NULL,
  `pr` SMALLINT UNSIGNED NOT NULL,
  `edema` TINYINT UNSIGNED NOT NULL,
  `weak_peripheral_pulse` TINYINT UNSIGNED NOT NULL,
  `lung_rales` TINYINT UNSIGNED NOT NULL,
  `systolic_murmur` TINYINT UNSIGNED NOT NULL,
  `diastolic_murmur` TINYINT UNSIGNED NOT NULL,
  `typical_chest_pain` TINYINT UNSIGNED NOT NULL,
  `dyspnea` TINYINT UNSIGNED NOT NULL,
  `function_class` TINYINT UNSIGNED NOT NULL,
  `atypical` TINYINT UNSIGNED NOT NULL,
  `nonanginal` TINYINT UNSIGNED NOT NULL,
  `lowth_ang` TINYINT UNSIGNED NOT NULL,
  `q_wave` TINYINT UNSIGNED NOT NULL,
  `st_elevation` TINYINT UNSIGNED NOT NULL,
  `st_depression` TINYINT UNSIGNED NOT NULL,
  `tinversion` TINYINT UNSIGNED NOT NULL,
  `lvh` TINYINT UNSIGNED NOT NULL,
  `poor_r_progression` TINYINT UNSIGNED NOT NULL,
  `bbb` TINYINT UNSIGNED NOT NULL,
  `fbs` DOUBLE NOT NULL,
  `cr` DOUBLE NOT NULL,
  `tg` DOUBLE NOT NULL,
  `ldl` SMALLINT UNSIGNED NOT NULL,
  `hdl` DOUBLE NOT NULL,
  `bun` SMALLINT UNSIGNED NOT NULL,
  `esr` SMALLINT UNSIGNED NOT NULL,
  `hb` DOUBLE NOT NULL,
  `k` DOUBLE NOT NULL,
  `na` SMALLINT UNSIGNED NOT NULL,
  `wbc` DOUBLE NOT NULL,
  `lymph` SMALLINT UNSIGNED NOT NULL,
  `neut` SMALLINT UNSIGNED NOT NULL,
  `plt` DOUBLE NOT NULL,
  `ef_tte` SMALLINT UNSIGNED NOT NULL,
  `region_rwma` TINYINT UNSIGNED NOT NULL,
  `vhd` TINYINT UNSIGNED NOT NULL,
  `lad` TINYINT UNSIGNED NOT NULL,
  `lcx` TINYINT UNSIGNED NOT NULL,
  `rca` TINYINT UNSIGNED NOT NULL,
  `cath` TINYINT UNSIGNED NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB;

-- Every prediction made through the API
CREATE TABLE IF NOT EXISTS predictions (
  id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  input_features JSON NOT NULL,
  cad_probability DOUBLE NOT NULL,
  lad_probability DOUBLE NOT NULL,
  lcx_probability DOUBLE NOT NULL,
  rca_probability DOUBLE NOT NULL,
  highest_vessel VARCHAR(3) NOT NULL,
  client_ip VARCHAR(45) NULL,
  INDEX ix_predictions_created_at (created_at)
) ENGINE=InnoDB;

-- Evaluation results from train_cad_xgboost.py (metrics.json)
CREATE TABLE IF NOT EXISTS model_metrics (
  target VARCHAR(8) NOT NULL PRIMARY KEY,   -- CAD, LAD, LCX, RCA
  label VARCHAR(64) NOT NULL,
  n_train INT NOT NULL,
  n_test INT NOT NULL,
  positive_rate DOUBLE,
  cv_auc_train DOUBLE,
  test_auc DOUBLE,
  test_accuracy DOUBLE,
  test_precision DOUBLE,
  test_recall DOUBLE,
  test_specificity DOUBLE,
  test_f1 DOUBLE,
  tn INT, fp INT, fn INT, tp INT,
  best_params JSON,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- feature_importances.csv in long format
CREATE TABLE IF NOT EXISTS feature_importance (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  target VARCHAR(8) NOT NULL,
  feature VARCHAR(64) NOT NULL,
  importance DOUBLE NOT NULL,
  rank_no INT NOT NULL,
  INDEX ix_fi_target (target)
) ENGINE=InnoDB;
