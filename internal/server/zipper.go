package server

import (
	"archive/zip"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

func createZip(dir, zipPath string) error {
	zipFile, err := os.Create(zipPath)
	if err != nil {
		return err
	}
	defer zipFile.Close()

	archive := zip.NewWriter(zipFile)
	defer archive.Close()

	return filepath.Walk(dir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()

		zipEntry, err := archive.Create(relPath)
		if err != nil {
			return err
		}

		_, err = io.Copy(zipEntry, file)

		return err
	})
}
