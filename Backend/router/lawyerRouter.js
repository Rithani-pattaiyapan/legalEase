import express from 'express';
import LawyerProfile from '../module/lawyer.module.js';
import Case from '../module/case.module.js';

const router = express.Router();


router.get('/', (req, res) => {
    res.send('Lawyer Router is working');
});

router.post('/process_update', async(req, res) => {
    try {
        const { caseId, updateText, lawyer_Id } = req.body;
        if (!caseId || !updateText || !lawyer_Id) {
            return res.status(400).json({ success: false, message: 'caseId, updateText and lawyer_Id are required' });
        }

        const caseData = await Case.findOne({_id : caseId,lawyer_id : lawyer_Id});
        if (!caseData) {
            return res.status(404).json({ success: false, message: 'Case not found' });
        }

        caseData.process_updates.push({
            update_text: updateText,
            updated_by: lawyer_Id,
            timestamp: new Date(),
        });

        await caseData.save();

        res.status(200).json({ 
            success: true, 
            message: 'Process update added successfully', 
            data: caseData 
        });
        } catch (error) {
            res.status(500).json({ 
                success: false, 
                message: 'Server Error', 
                error: error.message 
        });
        }
});

export default router;