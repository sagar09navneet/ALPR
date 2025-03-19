import express from 'express';
import swaggerUi from 'swagger-ui-express';
import swaggerJsdoc from 'swagger-jsdoc';
import multer from 'multer';
import cors from 'cors';
import Lens from 'chrome-lens-ocr';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import path from 'path';
import { unlink } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const port = 3000;
const lens = new Lens();

// Middleware
app.use(cors());
app.use(express.json());

// Multer configuration for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Swagger configuration
const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'OCR API',
      version: '1.0.0',
      description: 'API for performing OCR on images'
    },
    servers: [
      {
        url: `http://localhost:${port}`
      }
    ]
  },
  apis: ['./index.js']
};

const swaggerDocs = swaggerJsdoc(swaggerOptions);
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocs));



/**
 * @swagger
 * /ocr/upload:
 *   post:
 *     summary: Upload an image for OCR processing
 *     tags: [OCR]
 *     requestBody:
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               image:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: OCR results
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 text:
 *                   type: string
 */
app.post('/ocr/upload', upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded' });
      }
  
    try {
      const result = await lens.scanByFile(req.file.path);    
      await unlink(req.file.path);
      
      res.json(result);
    } catch (error) {
      try {
        await unlink(req.file.path);
      } catch (unlinkError) {
        console.error('Error deleting file:', unlinkError);
      }
      
      throw error;
    }
    

    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Error processing image' });
    }
  });
  
  /**
   * @swagger
   * /ocr/url:
   *   post:
   *     summary: Process OCR from image URL
   *     tags: [OCR]
   *     requestBody:
   *       required: true
   *       content:
   *         application/json:
   *           schema:
   *             type: object
   *             properties:
   *               url:
   *                 type: string
   *                 description: URL of the image to process
   *     responses:
   *       200:
   *         description: OCR results
   *         content:
   *           application/json:
   *             schema:
   *               type: object
   *               properties:
   *                 text:
   *                   type: string
   */
  app.post('/ocr/url', async (req, res) => {
    try {
      const { url } = req.body;
      if (!url) {
        return res.status(400).json({ error: 'Image URL is required' });
      }
  
      const result = await lens.scanByURL(url);
      res.json(result);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: 'Error processing image URL' });
    }
  });
  
  // Create uploads directory if it doesn't exist
  import { mkdir } from 'fs/promises';
  try {
    await mkdir('uploads');
  } catch (err) {
    if (err.code !== 'EEXIST') {
      console.error('Error creating uploads directory:', err);
    }
  }
  
  app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`Swagger documentation available at http://localhost:${port}/docs`);
  });

